package me.mikex86.scicore.tests;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.SciCore;
import me.mikex86.scicore.data.DatasetIterator;
import me.mikex86.scicore.nn.IModule;
import me.mikex86.scicore.nn.layers.Linear;
import me.mikex86.scicore.nn.layers.ReLU;
import me.mikex86.scicore.nn.layers.Softmax;
import me.mikex86.scicore.nn.optim.IOptimizer;
import me.mikex86.scicore.nn.optim.Sgd;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.utils.Pair;
import me.mikex86.scicore.utils.ShapeUtils;
import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarBuilder;
import me.tongfei.progressbar.ProgressBarStyle;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;

public class MNISTTest {

    @NotNull
    private static final Path MNIST_DIR = Path.of("mnist");

    private static void downloadMnist() throws IOException, InterruptedException {
        if (MNIST_DIR.toFile().exists()) {
            System.out.println("MNIST already downloaded");
            return;
        }
        HttpClient client = HttpClient.newHttpClient();
        List<String> urls = List.of(
                "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
        );
        Files.createDirectories(MNIST_DIR);

        for (String url : ProgressBar.wrap(urls, new ProgressBarBuilder()
                .setStyle(ProgressBarStyle.ASCII)
                .setTaskName("Downloading MNIST"))) {
            String filename = url.substring(url.lastIndexOf('/') + 1);
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .GET()
                    .build();
            Path path = MNIST_DIR.resolve(filename);
            client.send(request, HttpResponse.BodyHandlers.ofFile(path));

            // inflate gz files
            if (filename.endsWith(".gz")) {
                GZIPInputStream in = new GZIPInputStream(Files.newInputStream(path));
                Files.copy(in, path.resolveSibling(filename.substring(0, filename.length() - 3)));
            }
        }
    }

    private static final int BATCH_SIZE = 64;

    private static class MnistNet implements IModule {

        @NotNull
        private final ReLU act;

        @NotNull
        private final Linear fc1, fc2;

        @NotNull
        private final Softmax softmax;

        public MnistNet(@NotNull ISciCore sciCore) {
            this.act = new ReLU();
            this.fc1 = new Linear(sciCore, DataType.FLOAT32, 28 * 28, 128, true);
            this.fc2 = new Linear(sciCore, DataType.FLOAT32, 128, 10, true);
            this.softmax = new Softmax(sciCore, 1);
        }

        @Override
        public @NotNull ITensor forward(@NotNull ITensor input) {
            ITensor h = fc1.forward(input);
            h = act.forward(h);
            h = fc2.forward(h);
            return softmax.forward(h);
        }

        @Override
        public @NotNull List<ITensor> parameters() {
            return Stream.concat(fc1.parameters().stream(), fc2.parameters().stream()).collect(Collectors.toList());
        }

    }

    // A very simple tensor loader. This is not a stable deserializer and only works for the purposes of this test.
    private static ITensor readTensor(ISciCore sciCore, Path path) {
        try (RandomAccessFile file = new RandomAccessFile(path.toFile(), "r")) {
            // data-type string(32) ordinal, shape (n_dims: int64, dim1: int64, dim2: int64, ...), data
            byte[] buf = new byte[32];
            file.read(buf);
            String type = new String(buf).trim();
            DataType dataType = switch (type) {
                case "torch.float32":
                case "torch.float":
                    yield DataType.FLOAT32;
                case "torch.float64":
                    yield DataType.FLOAT64;
                case "torch.int32":
                case "torch.int":
                    yield DataType.INT32;
                case "torch.int64":
                    yield DataType.INT64;
                default:
                    throw new IllegalArgumentException("Unknown data type: " + type);
            };
            int nDims = Math.toIntExact(file.readLong());
            long[] shape = new long[nDims];
            for (int i = 0; i < nDims; i++) {
                shape[i] = file.readLong();
            }
            long nElements = ShapeUtils.getNumElements(shape);
            ITensor tensor = sciCore.zeros(dataType, shape);
            ByteBuffer buffer = ByteBuffer.allocateDirect(Math.toIntExact(nElements * dataType.getSize()))
                    .order(ByteOrder.LITTLE_ENDIAN);
            file.getChannel().read(buffer);
            buffer.flip();
            switch (dataType) {
                case FLOAT32 -> {
                    FloatBuffer floatBuffer = buffer.asFloatBuffer();
                    tensor.setContents(floatBuffer);
                }
                case FLOAT64 -> {
                    DoubleBuffer doubleBuffer = buffer.asDoubleBuffer();
                    tensor.setContents(doubleBuffer);
                }
                case INT32 -> {
                    IntBuffer intBuffer = buffer.asIntBuffer();
                    tensor.setContents(intBuffer);
                }
                case INT64 -> {
                    LongBuffer longBuffer = buffer.asLongBuffer();
                    tensor.setContents(longBuffer);
                }
                default -> throw new IllegalArgumentException("Unsupported data type: " + dataType);
            }
            return tensor;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        downloadMnist();

        ISciCore sciCore = new SciCore();
        sciCore.setBackend(ISciCore.BackendType.JVM);
        sciCore.seed(123);

        DatasetIterator trainIt = new DatasetIterator(BATCH_SIZE, new MnistDataSupplier(sciCore, true, false));
        DatasetIterator testIt = new DatasetIterator(1, new MnistDataSupplier(sciCore, false, false));

        MnistNet net = new MnistNet(sciCore);
        // set initial weights
        {
            ITensor fc1weight = readTensor(sciCore, MNIST_DIR.resolve("progress/fc1.weight_0.bin"));
            ITensor fc1bias = readTensor(sciCore, MNIST_DIR.resolve("progress/fc1.bias_0.bin"));
            net.fc1.getWeights().setContents(fc1weight);
            Objects.requireNonNull(net.fc1.getBias()).setContents(fc1bias);

            ITensor fc2weight = readTensor(sciCore, MNIST_DIR.resolve("progress/fc2.weight_0.bin"));
            ITensor fc2bias = readTensor(sciCore, MNIST_DIR.resolve("progress/fc2.bias_0.bin"));
            net.fc2.getWeights().setContents(fc2weight);
            Objects.requireNonNull(net.fc2.getBias()).setContents(fc2bias);
        }

        long nSteps = 33_000;
        int nTestSteps = 10000;

        float learningRate = 0.01f;

        IOptimizer optimizer = new Sgd(sciCore, learningRate, net.parameters());

        try (ProgressBar progressBar = new ProgressBar("Training", nSteps)) {
            for (long step = 0; step < nSteps; step++) {
                sciCore.getBackend().getOperationRecorder().resetRecording();
                Pair<ITensor, ITensor> batch = trainIt.next();
                ITensor X = batch.getFirst();
                ITensor Y = batch.getSecond();

                ITensor Y_pred = net.forward(X);
                // TODO: INVESTIGATE WHY THIS IS NOT WORKING WHEN USING reduceSum(1)
                ITensor loss = (Y_pred.minus(Y)).pow(2).reduceSum(-1).divided(Y_pred.getNumberOfElements());
                if (Float.isNaN(loss.elementAsFloat())) {
                    System.out.println("Loss is NaN");
                }
                if (step % 100 == 0) {
                    ITensor actualWeightsFc1 = net.fc1.getWeights();
                    ITensor actualBiasFc1 = Objects.requireNonNull(net.fc1.getBias());
                    ITensor expectedWeightsFc1 = readTensor(sciCore, MNIST_DIR.resolve("progress/fc1.weight_" + step + ".bin"));
                    ITensor expectedBiasFc1 = readTensor(sciCore, MNIST_DIR.resolve("progress/fc1.bias_" + step + ".bin"));
                    ITensor actualWeightsFc2 = net.fc2.getWeights();
                    ITensor actualBiasFc2 = Objects.requireNonNull(net.fc2.getBias());
                    ITensor expectedWeightsFc2 = readTensor(sciCore, MNIST_DIR.resolve("progress/fc2.weight_" + step + ".bin"));
                    ITensor expectedBiasFc2 = readTensor(sciCore, MNIST_DIR.resolve("progress/fc2.bias_" + step + ".bin"));

                    ITensor diffWeightsFc1 = actualWeightsFc1.minus(expectedWeightsFc1).reduceSum(-1);
                    ITensor diffBiasFc1 = actualBiasFc1.minus(expectedBiasFc1).reduceSum(-1);
                    ITensor diffWeightsFc2 = actualWeightsFc2.minus(expectedWeightsFc2).reduceSum(-1);
                    ITensor diffBiasFc2 = actualBiasFc2.minus(expectedBiasFc2).reduceSum(-1);

                    System.out.println("Step: " + step);
                    System.out.println("Loss: " + loss.elementAsFloat());
                    System.out.println("Diff weights fc1: " + diffWeightsFc1.elementAsFloat());
                    System.out.println("Diff bias fc1: " + diffBiasFc1.elementAsFloat());
                    System.out.println("Diff weights fc2: " + diffWeightsFc2.elementAsFloat());
                    System.out.println("Diff bias fc2: " + diffBiasFc2.elementAsFloat());
                }
//                if (step % 2000 == 0) {
//                    long nCorrect = 0;
//                    for (int i = 0; i < nTestSteps; i++) {
//                        sciCore.getBackend().getOperationRecorder().resetRecording();
//                        Pair<ITensor, ITensor> testBatch = testIt.next();
//                        ITensor testX = testBatch.getFirst();
//                        ITensor testY = testBatch.getSecond();
//                        ITensor testY_pred = net.forward(testX);
//                        ITensor pred_argMax = testY_pred.argmax(1);
//                        ITensor testY_argMax = testY.argmax(1);
//                        boolean correct = pred_argMax.equals(testY_argMax);
//                        if (correct) {
//                            nCorrect++;
//                        }
//                    }
//                    double accuracy = 0.0;
//                    if (nCorrect > 0) {
//                        accuracy = nCorrect / (double) nTestSteps;
//                    }
//                    System.out.println("\nStep: " + step + " Loss: " + loss.elementAsFloat() + " Accuracy: " + accuracy);
//                }
                IGraph graph = sciCore.getGraphUpTo(loss);
                optimizer.step(graph);

                progressBar.step();
                progressBar.setExtraMessage(String.format(Locale.US, "loss: %.5f", loss.elementAsFloat()));
            }
        }
        System.out.println("Done training");

    }

    private static class MnistDataSupplier implements Supplier<Pair<ITensor, ITensor>> {

        @NotNull
        private final RandomAccessFile imagesRAF;

        @NotNull
        private final RandomAccessFile labelsRAF;

        /**
         * List of (X, Y) pairs where X is the image and Y is the label.
         */
        @NotNull
        private final List<Pair<ITensor, ITensor>> samples;

        @Nullable
        private final Random random;

        private MnistDataSupplier(@NotNull ISciCore sciCore, boolean train, boolean shuffle) {
            if (shuffle) {
                this.random = new Random(123);
            } else {
                this.random = null;
            }

            Path imagesPath = MNIST_DIR.resolve(train ? "train-images-idx3-ubyte" : "t10k-images-idx3-ubyte");
            Path labelsPath = MNIST_DIR.resolve(train ? "train-labels-idx1-ubyte" : "t10k-labels-idx1-ubyte");

            try {
                imagesRAF = new RandomAccessFile(imagesPath.toFile(), "r");
                labelsRAF = new RandomAccessFile(labelsPath.toFile(), "r");


                int imagesMagic = imagesRAF.readInt();
                int nImages = imagesRAF.readInt();

                int labelsMagic = labelsRAF.readInt();
                int nLabels = labelsRAF.readInt();

                int height = imagesRAF.readInt();
                int width = imagesRAF.readInt();

                if (imagesMagic != 2051 || labelsMagic != 2049) {
                    throw new IOException("Invalid MNIST file");
                }

                if (nImages != nLabels) {
                    throw new IOException("Images and labels have different number of samples");
                }

                samples = new ArrayList<>(nImages);
                for (int i = 0; i < nImages; i++) {
                    int label = labelsRAF.readByte();
                    byte[] bytes = new byte[height * width];
                    imagesRAF.read(bytes);
                    float[] pixels = new float[width * height];
                    for (int j = 0; j < pixels.length; j++) {
                        pixels[j] = (bytes[j] & 0xFF) / 255.0f;
                    }
                    ITensor labelTensor = sciCore.zeros(DataType.FLOAT32, 10);
                    labelTensor.setFloatFlat(1, label);
                    samples.add(Pair.of(sciCore.array(pixels), labelTensor));
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

        }

        private int idx = 0;

        @NotNull
        public Pair<ITensor, ITensor> get() {
            Random random = this.random;
            if (random != null) {
                return samples.get(random.nextInt(samples.size()));
            } else {
                return samples.get(idx++ % samples.size());
            }
        }
    }
}
