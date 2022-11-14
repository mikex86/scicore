package me.mikex86.scicore.tests.mnist

import me.mikex86.scicore.ISciCore
import me.mikex86.scicore.SciCore
import me.mikex86.scicore.data.DatasetIterator
import me.mikex86.scicore.graph.scopedRecording
import me.mikex86.scicore.nn.optim.Sgd
import me.mikex86.scicore.tensor.DataType
import me.mikex86.scicore.utils.use
import me.tongfei.progressbar.ProgressBarBuilder
import me.tongfei.progressbar.ProgressBarStyle
import java.nio.file.Path
import java.util.*


private const val BATCH_SIZE = 32

// Recommended: -XX:+UseZGC -XX:+ExplicitGCInvokesConcurrent -XX:MaxGCPauseMillis=5
// The fact that the GC seems to not care about GC-ing memory handles because they are "small" on the Jvm heap (despite referencing large regions of native memory) is a bit concerning.
fun main() {
    val sciCore = SciCore()
    sciCore.setBackend(ISciCore.BackendType.CPU)
    sciCore.seed(123)

    val trainIt = DatasetIterator(BATCH_SIZE, MnistDataSupplier(sciCore, train = true, shuffle = false))
    val testIt = DatasetIterator(BATCH_SIZE, MnistDataSupplier(sciCore, train = false, shuffle = false))

    val net = MnistNet(sciCore)

    val nTrainSteps = 60_000L
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
                sciCore.backend.operationRecorder.scopedRecording {
                    val batch = trainIt.next()
                    batch.use { x, y ->
                        lossValue = net.forward(x)
                            .use { yPred -> yPred.minus(y) }
                            .use { diff -> diff.pow(2f) }
                            .use { diffSquared -> diffSquared.reduceSum(-1) }
                            .use { sum -> sum.divide(BATCH_SIZE.toFloat()) }
                            .use { loss -> optimizer.step(loss); loss.elementAsDouble() }
                    }
                }
                progressBar.step()
                progressBar.extraMessage = String.format(Locale.US, "loss: %.5f", lossValue)
            }
        }
    val end = System.currentTimeMillis()
    println("Training time: " + (end - start) / 1000.0 + "s")
    println("Final loss value: $lossValue")

    System.out.flush()

    println("Start testing...")

    var correct = 0

    ProgressBarBuilder()
        .setTaskName("Testing")
        .setInitialMax(nTestSteps)
        .setStyle(ProgressBarStyle.UNICODE_BLOCK)
        .setUpdateIntervalMillis(100)
        .build().use { progressBar ->
            for (testStep in 0 until nTestSteps) {
                sciCore.backend.operationRecorder.scopedRecording {
                    val batch = testIt.next()
                    batch.use { x, y ->
                        net.forward(x)
                            .use { yPred -> yPred.argmax(1) }
                            .use { yPredMax ->
                                correct += y.argmax(1)
                                    .use { yMax -> yPredMax.compareElements(yMax) }
                                    .use { yCmpBool -> yCmpBool.cast(DataType.INT32) }
                                    .use { yCmpInt -> yCmpInt.reduceSum(-1) }
                                    .use { yCmpSum -> yCmpSum.elementAsInt() }
                            }
                    }
                }
                progressBar.step()
                progressBar.extraMessage =
                    String.format(Locale.US, "accuracy: %.5f", correct.toFloat() / testStep / BATCH_SIZE)
            }
        }
    println("Final Accuracy: " + correct.toFloat() / nTestSteps / BATCH_SIZE)
    net.save(Path.of("mnist.scm"))
}