package me.mikex86.scicore.tests.makemore

import me.mikex86.scicore.ISciCore
import me.mikex86.scicore.SciCore
import me.mikex86.scicore.data.DatasetIterator
import me.mikex86.scicore.graph.scopedRecording
import me.mikex86.scicore.nn.IModule
import me.mikex86.scicore.nn.act.Tanh
import me.mikex86.scicore.nn.layers.Linear
import me.mikex86.scicore.nn.optim.Sgd
import me.mikex86.scicore.tensor.DataType
import me.mikex86.scicore.tensor.ITensor
import me.mikex86.scicore.tensor.LazyTensor
import me.mikex86.scicore.tensor.unaryMinus
import me.mikex86.scicore.utils.use
import me.tongfei.progressbar.ProgressBarBuilder
import me.tongfei.progressbar.ProgressBarStyle
import java.util.*

private const val BATCH_SIZE = 32
private const val BLOCK_SIZE = 3
private const val EMBEDDING_SIZE = 10
private const val N_HIDDEN = 128
private const val VOCAB_SIZE = 26 + 1 // 26 letters + 1 padding char

private const val N_TRAINING_STEPS = 200_000L
private const val LEARNING_RATE = 0.01f

fun main() {
    val sciCore = SciCore()
    sciCore.setBackend(ISciCore.BackendType.CPU)
    sciCore.seed(123)

    val trainIt = DatasetIterator(BATCH_SIZE, NamesDatasetSupplier(sciCore, BLOCK_SIZE, true))
    val testIt = DatasetIterator(BATCH_SIZE, NamesDatasetSupplier(sciCore, BLOCK_SIZE, false))

    val net = MakeMoreNet(sciCore)

    val optimizer = Sgd(sciCore, LEARNING_RATE, net.parameters())
    var lossValue = -1.0

    println("Start training...")
    val start = System.currentTimeMillis()
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
                            .use { counts -> counts / counts.reduceSum(1, true) }
                            .use { probs -> -probs[sciCore.arange(0, 32, 1, DataType.INT64), y] }
                            .use { probsAssignedToCorrectLabels ->
                                probsAssignedToCorrectLabels.mean()
                            }.use { loss ->
                                optimizer.step(loss); loss.elementAsDouble()
                            }
                    }
                }
                progressBar.step()
                progressBar.extraMessage = String.format(Locale.US, "loss: %.5f", lossValue)
            }
        }
    val end = System.currentTimeMillis()
    println("Training time: " + (end - start) / 1000.0 + "s")
    println("Final loss value: $lossValue")

}

class MakeMoreNet(private val sciCore: SciCore) : IModule {

    private val embedding = sciCore.uniform(DataType.FLOAT32, VOCAB_SIZE.toLong(), EMBEDDING_SIZE.toLong())
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
                        BATCH_SIZE.toLong(),
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