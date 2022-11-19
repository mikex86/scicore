package me.mikex86.scicore.tests.makemore

import me.mikex86.matplotlib.jplot.JPlot
import me.mikex86.scicore.ISciCore
import me.mikex86.scicore.SciCore
import me.mikex86.scicore.data.DatasetIterator
import me.mikex86.scicore.graph.GraphExecutor
import me.mikex86.scicore.graph.scopedRecording
import me.mikex86.scicore.nn.IModule
import me.mikex86.scicore.nn.act.Tanh
import me.mikex86.scicore.nn.layers.Linear
import me.mikex86.scicore.nn.optim.Sgd
import me.mikex86.scicore.profiling.Profiler
import me.mikex86.scicore.tensor.DataType
import me.mikex86.scicore.tensor.ITensor
import me.mikex86.scicore.tensor.LazyTensor
import me.mikex86.scicore.tensor.unaryMinus
import me.mikex86.scicore.utils.use
import me.tongfei.progressbar.ProgressBarBuilder
import me.tongfei.progressbar.ProgressBarStyle
import java.awt.Color
import java.nio.file.Path
import java.util.*

private const val BATCH_SIZE = 32
private const val BLOCK_SIZE = 3
private const val EMBEDDING_SIZE = 10
private const val N_HIDDEN = 200
private const val VOCAB_SIZE = 26 + 1 // 26 letters + 1 padding char

private const val N_TRAINING_STEPS = 200_000L
private const val INITIAL_LEARNING_RATE = 0.1f
private const val END_LEARNING_RATE = 0.05f

fun main() {
    val sciCore = SciCore()
    sciCore.setBackend(ISciCore.BackendType.CPU)
    sciCore.seed(123)

    val namesSupplier = NamesDatasetSupplier(sciCore, BLOCK_SIZE, training = true, shuffle = true)
    val trainIt = DatasetIterator(BATCH_SIZE, namesSupplier)

    val net = MakeMoreNet(sciCore)

    val optimizer = Sgd(sciCore, INITIAL_LEARNING_RATE, END_LEARNING_RATE, N_TRAINING_STEPS, net.parameters())
    var lossValue = -1.0


    // loss on dataset
    sciCore.backend.operationRecorder.scopedRecording {
        val x = namesSupplier.x
        val y = namesSupplier.y
        val loss = net(x)
            // cross entropy loss
            .use { logits -> logits.exp() }
            .use { counts -> counts / counts.reduceSum(1, true) }
            .use { probs ->
                sciCore.arange(0, y.shape[0], 1, DataType.INT64)
                    .use { idx -> probs[idx, y] }
            }
            .use { probsAssignedToCorrectLabels ->
                -probsAssignedToCorrectLabels.log().mean().elementAsDouble()
            }
        println("Loss on dataset before training: $loss")
    }

    println("Start training...")
    val start = System.currentTimeMillis()
    val losses = FloatArray(N_TRAINING_STEPS.toInt())
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
                            .use { logits -> logits.exp() }
                            .use { counts ->
                                counts.reduceSum(1, true)
                                    .use { totalCounts -> counts / totalCounts }
                            }
                            .use { probs ->
                                sciCore.arange(0, 32, 1, DataType.INT64)
                                    .use { idx -> probs[idx, y] }
                            }
                            .use { probsAssignedToCorrectLabels ->
                                probsAssignedToCorrectLabels.log()
                            }.use { logProbs ->
                                logProbs.mean()
                            }
                            .use { logProbsMean ->
                                -logProbsMean
                            }.use { negativeLogLikelyHood ->
                                val loss = negativeLogLikelyHood.elementAsDouble()
                                optimizer.step(negativeLogLikelyHood)
                                loss
                            }
                    }
                }
                progressBar.step()
                progressBar.extraMessage = String.format(Locale.US, "loss: %.5f", lossValue)
                losses[step.toInt()] = lossValue.toFloat()
            }
        }
    val end = System.currentTimeMillis()
    println("Training time: " + (end - start) / 1000.0 + "s")
    net.save(Path.of("makemore.scm"))

    // plot losses
    val plot = JPlot()
    plot.plot(losses, Color(26, 188, 156), false)
    plot.setXLabel("Step")
    plot.setYLabel("Loss (log)")
    plot.save(Path.of("makemore_loss.png"))

    // loss on dataset
    sciCore.backend.operationRecorder.scopedRecording {
        val x = namesSupplier.x
        val y = namesSupplier.y
        val loss = net(x)
            // cross entropy loss
            .use { logits -> logits.exp() }
            .use { counts -> counts / counts.reduceSum(1, true) }
            .use { probs -> probs[sciCore.arange(0, y.shape[0], 1, DataType.INT64), y] }
            .use { probsAssignedToCorrectLabels ->
                -probsAssignedToCorrectLabels.log().mean().elementAsDouble()
            }
        println("Loss on dataset: $loss")
    }

    // Print number of executed operations
    println("Number of performed operations: " + GraphExecutor.getNumOperations())
}

class MakeMoreNet(sciCore: SciCore) : IModule {

    private val embedding = (sciCore.gaussian(
        DataType.FLOAT32,
        VOCAB_SIZE.toLong(),
        EMBEDDING_SIZE.toLong()
    ).minus(0.5f) as LazyTensor).result()

    private val fc1 = Linear(sciCore, DataType.FLOAT32, (EMBEDDING_SIZE * BLOCK_SIZE).toLong(), N_HIDDEN.toLong(), true)
    private val fc2 = Linear(sciCore, DataType.FLOAT32, N_HIDDEN.toLong(), VOCAB_SIZE.toLong(), true)
    private val act = Tanh()

    override fun forward(input: ITensor): ITensor {
        // input: (batchSize, blockSize)
        val logits = embedding[input] // embeddingsForSequence: (batchSize, blockSize, embeddingSize)
            .use { embeddingsForSequence ->
                // embeddingsForSequenceFlat: (batchSize, blockSize * embeddingSize)
                embeddingsForSequence.getReshapedView(
                    longArrayOf(
                        -1,
                        (EMBEDDING_SIZE * BLOCK_SIZE).toLong()
                    )
                )
            }
            .use { embeddingsForSequenceFlat ->
                fc1(embeddingsForSequenceFlat) // fc1Output: (batchSize, nHidden)
            }
            .use { fc1Output ->
                act(fc1Output) // fc1WithAct: (batchSize, nHidden)
            }
            .use { fc1WithAct ->
                fc2(fc1WithAct) // logits: (batchSize, vocabSize)
            }
        return logits
    }

    override fun subModules(): List<IModule> {
        return listOf(fc1, fc2)
    }

    override fun parameters(): List<ITensor> {
        return listOf(fc1.parameters(), fc2.parameters())
            .flatten()
            .toMutableList()
            .apply { add(embedding) }
    }

}