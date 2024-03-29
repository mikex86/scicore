package me.mikex86.scicore.tests.mnist

import me.mikex86.scicore.ISciCore
import me.mikex86.scicore.tensor.DataType
import me.mikex86.scicore.tensor.ITensor
import java.io.IOException
import java.io.RandomAccessFile
import java.util.*
import java.util.function.Supplier
import kotlin.collections.ArrayList
import me.tongfei.progressbar.ProgressBar
import me.tongfei.progressbar.ProgressBarBuilder
import me.tongfei.progressbar.ProgressBarStyle
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.file.Files
import java.nio.file.Path
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

class MnistDataSupplier(sciCore: ISciCore, train: Boolean, shuffle: Boolean) : Supplier<Pair<ITensor, ITensor>> {

    private val imagesRAF: RandomAccessFile
    private val labelsRAF: RandomAccessFile

    val x: ITensor
    val y: ITensor
    private val random: Random?
    private var idx = 0L

    companion object {
        init {
            downloadMnist()
        }
    }

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
            x = sciCore.zeros(DataType.FLOAT32, nImages.toLong(), height.toLong() * width.toLong())
            y = sciCore.zeros(DataType.FLOAT32, nImages.toLong(), 10)

            val imageData = ByteBuffer.allocateDirect(height * width * 4).order(ByteOrder.LITTLE_ENDIAN)
            for (i in 0 until nImages) {
                val label = labelsRAF.readByte()
                val bytes = ByteArray(height * width)
                imagesRAF.read(bytes)

                for (j in 0 until height * width) {
                    imageData.putFloat((bytes[j].toInt() and 0xFF) / 255.0f)
                }
                imageData.flip()

                x.setContents(longArrayOf(i.toLong()), imageData)
                y.setFloat(1.0f, i.toLong(), label.toLong())
            }
        } catch (e: IOException) {
            throw RuntimeException(e)
        }
    }

    override fun get(): Pair<ITensor, ITensor> {
        val random = random
        return if (random != null) {
            Pair(
                x.getView(random.nextLong(x.shape[0])),
                y.getView(random.nextLong(y.shape[0]))
            )
        } else {
            Pair(
                x.getView(idx),
                y.getView(idx)
            ).also {
                idx++
                idx %= x.shape[0]
            }
        }
    }
}