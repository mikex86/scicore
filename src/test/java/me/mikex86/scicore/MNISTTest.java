package me.mikex86.scicore;

import me.mikex86.scicore.data.DatasetIterator;
import me.mikex86.scicore.nn.IModule;
import me.mikex86.scicore.nn.layers.Linear;
import me.mikex86.scicore.nn.layers.Sigmoid;
import me.mikex86.scicore.nn.layers.Softmax;
import me.mikex86.scicore.nn.optim.IOptimizer;
import me.mikex86.scicore.nn.optim.Sgd;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.utils.Pair;
import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarBuilder;
import me.tongfei.progressbar.ProgressBarStyle;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
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

    private static final int BATCH_SIZE = 32;

    private static class MnistNet implements IModule {

        @NotNull
        private final Sigmoid act;

        @NotNull
        private final Linear f1, f2;

        @NotNull
        private final Softmax f3;

        public MnistNet(@NotNull ISciCore sciCore) {
            this.act = new Sigmoid();
            this.f1 = new Linear(sciCore, DataType.FLOAT32, 28 * 28, 128, true);
            this.f2 = new Linear(sciCore, DataType.FLOAT32, 128, 10, true);
            this.f3 = new Softmax(sciCore, 1);
        }

        @Override
        public @NotNull ITensor forward(@NotNull ITensor input) {
            ITensor h1 = f1.forward(input);
            ITensor h2 = act.forward(h1);
            ITensor h3 = f2.forward(h2);
            ITensor h4 = act.forward(h3);
            return f3.forward(h4);
        }

        @Override
        public @NotNull List<ITensor> parameters() {
            return Stream.concat(f1.parameters().stream(), f2.parameters().stream()).collect(Collectors.toList());
        }

    }

    public static void main(String[] args) throws IOException, InterruptedException {
        // TODO: THIS DOESN'T LEARN... YIKES
        downloadMnist();

        ISciCore sciCore = new SciCore();
        sciCore.setBackend(ISciCore.BackendType.JVM);

        DatasetIterator trainIt = new DatasetIterator(BATCH_SIZE, new MnistDataSupplier(sciCore, true));
        DatasetIterator testIt = new DatasetIterator(BATCH_SIZE, new MnistDataSupplier(sciCore, false));

        MnistNet bobNet = new MnistNet(sciCore);

        long nSteps = 1000;

        float learningRate = 0.1f;

        IOptimizer optimizer = new Sgd(sciCore, learningRate, bobNet.parameters());

        try (ProgressBar progressBar = new ProgressBar("Training", nSteps)) {
            for (long step = 0; step < nSteps; step++) {
                sciCore.getBackend().getOperationRecorder().resetRecording();
                Pair<ITensor, ITensor> batch = trainIt.next();
                ITensor X = batch.getFirst();
                ITensor Y = batch.getSecond();

                ITensor Y_pred = bobNet.forward(X);
                // TODO: INVESTIGATE WHY THIS IS NOT WORKING WHEN USING reduceSum(1)
                ITensor loss = (Y_pred.minus(Y)).pow(2).reduceSum(0).reduceSum(0).divided(X.getShape()[0]);

                IGraph graph = sciCore.getGraphUpTo(loss);
                optimizer.step(graph);

                progressBar.step();
                progressBar.setExtraMessage(String.format("Loss: %.3f", loss.elementAsFloat()));
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

        private final Random random;

        private MnistDataSupplier(@NotNull ISciCore sciCore, boolean train) {
            this.random = new Random();

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

        @NotNull
        public Pair<ITensor, ITensor> get() {
            return samples.get(random.nextInt(samples.size()));
        }
    }
}
