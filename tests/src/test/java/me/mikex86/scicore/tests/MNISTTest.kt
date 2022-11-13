package me.mikex86.scicore.tests

import me.mikex86.scicore.ISciCore
import me.mikex86.scicore.SciCore
import me.mikex86.scicore.data.DatasetIterator
import me.mikex86.scicore.graph.scopedRecording
import me.mikex86.scicore.nn.IModule
import me.mikex86.scicore.nn.layers.Linear
import me.mikex86.scicore.nn.layers.ReLU
import me.mikex86.scicore.nn.layers.Softmax
import me.mikex86.scicore.nn.optim.Sgd
import me.mikex86.scicore.tensor.DataType
import me.mikex86.scicore.tensor.ITensor
import me.mikex86.scicore.utils.Pair
import me.mikex86.scicore.utils.use
import me.tongfei.progressbar.ProgressBar
import me.tongfei.progressbar.ProgressBarBuilder
import me.tongfei.progressbar.ProgressBarStyle
import java.io.IOException
import java.io.RandomAccessFile
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import java.nio.file.Path
import java.util.*
import java.util.function.Supplier
import java.util.zip.GZIPInputStream


private val MNIST_DIR = Path.of("mnist")

@Throws(IOException::class, InterruptedException::class)
private fun downloadMnist() {
    if (MNIST_DIR.toFile().exists()) {
        println("MNIST already downloaded")
        return
    }
    val client = HttpClient.newHttpClient()
    val urls = listOf(
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
    )
    Files.createDirectories(MNIST_DIR)
    for (url in ProgressBar.wrap(
        urls, ProgressBarBuilder()
            .setStyle(ProgressBarStyle.ASCII)
            .setTaskName("Downloading MNIST")
    )) {
        val filename = url.substring(url.lastIndexOf('/') + 1)
        val request = HttpRequest.newBuilder()
            .uri(URI.create(url))
            .GET()
            .build()
        val path = MNIST_DIR.resolve(filename)
        client.send(request, HttpResponse.BodyHandlers.ofFile(path))

        // inflate gz files
        if (filename.endsWith(".gz")) {
            val `in` = GZIPInputStream(Files.newInputStream(path))
            Files.copy(`in`, path.resolveSibling(filename.substring(0, filename.length - 3)))
        }
    }
}

private const val BATCH_SIZE = 32

// Recommended: -XX:+UseZGC -XX:+ExplicitGCInvokesConcurrent -XX:MaxGCPauseMillis=5
// The fact that the GC seems to not care about GC-ing memory handles because they are "small" on the Jvm heap (despite referencing large regions of native memory) is a bit concerning.
fun main() {
    downloadMnist()

    val sciCore = SciCore()
    sciCore.setBackend(ISciCore.BackendType.CPU)
    sciCore.seed(123)

    val trainIt = DatasetIterator(BATCH_SIZE, MnistDataSupplier(sciCore, train = true, shuffle = false))
    val testIt = DatasetIterator(BATCH_SIZE, MnistDataSupplier(sciCore, train = false, shuffle = false))

    val net = MnistNet(sciCore)

    val nTrainSteps = 20_000L
    val nTestSteps = 10_000L
    val learningRate = 0.01f

    val optimizer = Sgd(sciCore, learningRate, net.parameters())

    println("Start training...")

    val start = System.currentTimeMillis()
    var lossValue = -1.0

    ProgressBarBuilder()
        .setTaskName("Training")
        .setInitialMax(nTrainSteps)
        .setStyle(ProgressBarStyle.UNICODE_BLOCK)
        .setUpdateIntervalMillis(100)
        .build().use { progressBar ->
            for (step in 0 until nTrainSteps) {
                sciCore.backend.operationRecorder.resetRecording()
//                sciCore.backend.operationRecorder.scopedRecording {
                    val batch = trainIt.next()
                    batch.use { x, y ->
                        lossValue = net.forward(x)
                            .use { yPred -> yPred.minus(y) }
                            .use { diff -> diff.pow(2f) }
                            .use { diffSquared -> diffSquared.reduceSum(-1) }
                            .use { sum -> sum.divide(BATCH_SIZE.toFloat()) }
                            .use { loss -> optimizer.step(loss); loss.elementAsDouble() }
                    }
//                }
                progressBar.step()
                progressBar.extraMessage = String.format(Locale.US, "loss: %.5f", lossValue)
            }
        }
    val end = System.currentTimeMillis()
    println("Training time: " + (end - start) / 1000.0 + "s")
    println("Final loss value: $lossValue")

    System.out.flush()

//    println("Start testing...")
//
//    var correct = 0
//
//    ProgressBarBuilder()
//        .setTaskName("Testing")
//        .setInitialMax(nTestSteps)
//        .setStyle(ProgressBarStyle.UNICODE_BLOCK)
//        .setUpdateIntervalMillis(100)
//        .build().use { progressBar ->
//            for (testStep in 0 until nTestSteps) {
//                sciCore.backend.operationRecorder.scopedRecording {
//                    val batch = testIt.next()
//                    batch.use { x, y ->
//                        net.forward(x)
//                            .use { yPred -> yPred.argmax(1) }
//                            .use { yPredMax ->
//                                correct += y.argmax(1)
//                                    .use { yMax -> yPredMax.compareElements(yMax) }
//                                    .use { yCmpBool -> yCmpBool.cast(DataType.INT32) }
//                                    .use { yCmpInt -> yCmpInt.reduceSum(-1) }
//                                    .use { yCmpSum -> yCmpSum.elementAsInt() }
//                            }
//                    }
//                }
//                progressBar.step()
//                progressBar.extraMessage =
//                    String.format(Locale.US, "accuracy: %.5f", correct.toFloat() / testStep / BATCH_SIZE)
//            }
//        }
//    println("Final Accuracy: " + correct.toFloat() / nTestSteps / BATCH_SIZE)
}

class MnistNet(sciCore: ISciCore) : IModule {

    private val act = ReLU()
    private val fc1 = Linear(sciCore, DataType.FLOAT32, (28 * 28).toLong(), 128, true)
    private val fc2 = Linear(sciCore, DataType.FLOAT32, 128, 10, true)
    private val softmax = Softmax(sciCore, 1)

    override fun forward(input: ITensor): ITensor {
        return fc1.forward(input)
            .use { h -> act.forward(h) }
            .use { h -> fc2.forward(h) }
            .use { h -> softmax.forward(h) }
    }

    override fun parameters(): List<ITensor> {
        return collectParameters(fc1, fc2)
    }
}

class MnistDataSupplier(sciCore: ISciCore, train: Boolean, shuffle: Boolean) : Supplier<Pair<ITensor, ITensor>> {

    private val imagesRAF: RandomAccessFile
    private val labelsRAF: RandomAccessFile

    /**
     * List of (X, Y) pairs where X is the image and Y is the label.
     */
    private val samples: MutableList<Pair<ITensor, ITensor>>
    private val random: Random?
    private var idx = 0

    init {
        random = if (shuffle) {
            Random(123)
        } else {
            null
        }
        val imagesPath = MNIST_DIR.resolve(if (train) "train-images-idx3-ubyte" else "t10k-images-idx3-ubyte")
        val labelsPath = MNIST_DIR.resolve(if (train) "train-labels-idx1-ubyte" else "t10k-labels-idx1-ubyte")
        try {
            imagesRAF = RandomAccessFile(imagesPath.toFile(), "r")
            labelsRAF = RandomAccessFile(labelsPath.toFile(), "r")
            val imagesMagic = imagesRAF.readInt()
            val nImages = imagesRAF.readInt()
            val labelsMagic = labelsRAF.readInt()
            val nLabels = labelsRAF.readInt()
            val height = imagesRAF.readInt()
            val width = imagesRAF.readInt()
            if (imagesMagic != 2051 || labelsMagic != 2049) {
                throw IOException("Invalid MNIST file")
            }
            if (nImages != nLabels) {
                throw IOException("Images and labels have different number of samples")
            }
            samples = ArrayList(nImages)
            for (i in 0 until nImages) {
                val label = labelsRAF.readByte().toInt()
                val bytes = ByteArray(height * width)
                imagesRAF.read(bytes)
                val pixels = FloatArray(width * height)
                for (j in pixels.indices) {
                    pixels[j] = (bytes[j].toInt() and 0xFF) / 255.0f
                }
                val labelTensor = sciCore.zeros(DataType.FLOAT32, 10)
                labelTensor.setFloatFlat(1f, label.toLong())
                samples.add(Pair.of(sciCore.array(pixels), labelTensor))
            }
        } catch (e: IOException) {
            throw RuntimeException(e)
        }
    }

    override fun get(): Pair<ITensor, ITensor> {
        val random = random
        return if (random != null) {
            samples[random.nextInt(samples.size)]
        } else {
            samples[idx++ % samples.size]
        }
    }
}