package me.mikex86.scicore.tests.mnist

import me.mikex86.matplotlib.jplot.JPlot
import me.mikex86.scicore.ISciCore
import me.mikex86.scicore.SciCore
import me.mikex86.scicore.data.DatasetIterator
import me.mikex86.scicore.graph.scopedRecording
import me.mikex86.scicore.nn.optim.Adam
import me.mikex86.scicore.nn.optim.Sgd
import me.mikex86.scicore.tensor.DataType
import me.mikex86.scicore.utils.use
import me.tongfei.progressbar.ProgressBarBuilder
import me.tongfei.progressbar.ProgressBarStyle
import java.awt.Color
import java.nio.file.Path
import java.util.*


private const val BATCH_SIZE = 32

private const val N_TRAINING_STEPS = 60_000L
private const val N_TEST_STEPS = 20_000L
private const val LEARNING_RATE = 0.01f
private const val RMS_DECAY_FACTOR = 0.999f
private const val MOMENTUM_FACTOR = 0.9f
private const val DAMPENING_FACTOR = 0.1f

// Recommended: -XX:+UseZGC -XX:+ExplicitGCInvokesConcurrent -XX:MaxGCPauseMillis=5
// The fact that the GC seems to not care about GC-ing memory handles because they are "small" on the Jvm heap (despite referencing large regions of native memory) is a bit concerning.
fun main() {
    val sciCore = SciCore()
    sciCore.addBackend(ISciCore.BackendType.CPU)
//    sciCore.addBackend(ISciCore.BackendType.CUDA)
    sciCore.seed(123)

    val trainSupplier = MnistDataSupplier(sciCore, train = true, shuffle = false)
    val testSupplier = MnistDataSupplier(sciCore, train = false, shuffle = false)

    val trainIt = DatasetIterator(BATCH_SIZE, trainSupplier)
    val testIt = DatasetIterator(BATCH_SIZE, testSupplier)

    val net = MnistNet(sciCore)

    val optimizer = Sgd(sciCore, LEARNING_RATE, net.parameters())
//    val optimizer = SgdWithMomentum(sciCore, LEARNING_RATE, MOMENTUM_FACTOR, DAMPENING_FACTOR, net.parameters())
//    val optimizer = Adam(sciCore, LEARNING_RATE, MOMENTUM_FACTOR, RMS_DECAY_FACTOR, net.parameters())

    println("Start training...")

    val start = System.currentTimeMillis()
    var lossValue = -1.0

    val losses = sciCore.zeros(DataType.FLOAT32, N_TRAINING_STEPS)

    ProgressBarBuilder()
        .setTaskName("Training")
        .setInitialMax(N_TRAINING_STEPS)
        .setStyle(ProgressBarStyle.UNICODE_BLOCK)
        .setUpdateIntervalMillis(100)
        .build().use { progressBar ->
            for (step in 0 until N_TRAINING_STEPS) {
                sciCore.backend.operationRecorder.scopedRecording {
                    val batch = trainIt.next()
                    batch.use { x, y ->
                        lossValue = net(x)
                            .use { yPred -> yPred.minus(y) }
                            .use { diff -> diff.pow(2f) }
                            .use { diffSquared -> diffSquared.reduceSum(-1) }
                            .use { sum -> sum.divide(BATCH_SIZE.toFloat()) }
                            .use { loss ->
                                optimizer.step(loss)
                                loss.elementAsDouble()
                            }
                    }
                }
                progressBar.step()
                progressBar.extraMessage = String.format(Locale.US, "loss: %.5f", lossValue)
                losses.setFloat(lossValue.toFloat(), step)
            }
        }
    val end = System.currentTimeMillis()
    println("Training time: " + (end - start) / 1000.0 + "s")
    // loss on dataset
    sciCore.backend.operationRecorder.scopedRecording {
        val loss = net(trainSupplier.x)
            .use { yPred -> yPred.minus(trainSupplier.y) }
            .use { diff -> diff.pow(2f) }
            .use { diffSquared -> diffSquared.reduceSum(-1) }
            .use { loss ->
                loss.elementAsDouble()
            }
        println("Loss on training set after training: $loss")
    }
    println("Examples per second: " + N_TRAINING_STEPS * BATCH_SIZE / ((end - start) / 1000.0))

    val jplot = JPlot()
    jplot.setName("MNIST training")

    jplot.setXLabel("Step")
    jplot.setYLabel("Loss")
    jplot.setBeginY(-0.01f)
//    val avgLosses = losses.view(-1, 100).mean(1)
    val lossesArray = FloatArray(losses.numberOfElements.toInt()) { losses.getFloat(it.toLong()) }
    jplot.plot(lossesArray, Color(46, 204, 113), true)
    jplot.save(Path.of("mnist_loss.png"))

    System.out.flush()

    println("Start testing...")

    var correct = 0

    ProgressBarBuilder()
        .setTaskName("Testing")
        .setInitialMax(N_TEST_STEPS)
        .setStyle(ProgressBarStyle.UNICODE_BLOCK)
        .setUpdateIntervalMillis(100)
        .build().use { progressBar ->
            for (testStep in 0 until N_TEST_STEPS) {
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
    println("Test Accuracy: " + correct.toFloat() / N_TEST_STEPS / BATCH_SIZE)
    net.save(Path.of("mnist.scm"))
}