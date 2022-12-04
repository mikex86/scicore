package me.mikex86.scicore.tests.makemore.rnn

import me.mikex86.scicore.ISciCore
import me.mikex86.scicore.SciCore
import me.mikex86.scicore.data.DatasetIterator
import me.mikex86.scicore.graph.scopedRecording
import me.mikex86.scicore.nn.IModule
import me.mikex86.scicore.nn.act.Tanh
import me.mikex86.scicore.nn.layers.Linear
import me.mikex86.scicore.nn.optim.Sgd
import me.mikex86.scicore.ranges.ALL
import me.mikex86.scicore.tensor.DataType
import me.mikex86.scicore.tensor.ITensor
import me.mikex86.scicore.tensor.get
import me.mikex86.scicore.tests.makemore.NamesLeftShiftDatasetSupplier
import me.mikex86.scicore.utils.use
import me.tongfei.progressbar.ProgressBarBuilder
import me.tongfei.progressbar.ProgressBarStyle
import java.util.*

private const val VOCAB_SIZE = 26L + 1 // 26 letters + 1 start/end char
private const val EMBEDDING_SIZE = 32L
private const val HIDDEN_SIZE = 64L

private const val N_PREDICTIONS = 10L
private const val PREDICTION_MAX_WORD_LEN = 20L


private const val BATCH_SIZE = 32
private const val N_TRAINING_STEPS = 200_000L

fun main() {
    val sciCore = SciCore()
    sciCore.setBackend(ISciCore.BackendType.CPU)
    sciCore.seed(123)

    val net = MakeMoreRnnNet(sciCore)

    val trainIt = DatasetIterator(BATCH_SIZE, NamesLeftShiftDatasetSupplier(sciCore, training = true, shuffle = true))
    val testIt = DatasetIterator(BATCH_SIZE, NamesLeftShiftDatasetSupplier(sciCore, training = false, shuffle = false))

    val optimizer = Sgd(sciCore, 0.05f, net.parameters())

    val lastTime = System.currentTimeMillis()

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
                        val lossValue = net(x)
                            .use { logits ->
                                sciCore.crossEntropy(
                                    logits.view(-1L, logits.shape.last()),
                                    y.view(-1L), -1L
                                )
                            }
                            .use { loss ->
                                optimizer.step(loss)
                                loss.elementAsDouble()
                            }
                        val now = System.currentTimeMillis()
                        val elapsed = now - lastTime
                        val stepsPerSecond = step / (elapsed / 1000.0)
                        progressBar.step()
                        progressBar.extraMessage =
                            String.format(Locale.US, "loss: %.5f, steps/s: %.2f", lossValue, stepsPerSecond)
                    }
                }
            }
        }
}

class MakeMoreRnnNet(private val sciCore: SciCore) : IModule {

    private val embedding = sciCore.gaussian(
        DataType.FLOAT32,
        VOCAB_SIZE,
        EMBEDDING_SIZE
    )

    // Deviation from Mikolov et al. 2010, initial hidden state is learned, not zero
    private val start = sciCore.gaussian(
        DataType.FLOAT32,
        1,
        HIDDEN_SIZE
    )

    private val rnnCell = Linear(sciCore, DataType.FLOAT32, EMBEDDING_SIZE + HIDDEN_SIZE, HIDDEN_SIZE, true)
    private val lmHead = Linear(sciCore, DataType.FLOAT32, HIDDEN_SIZE, VOCAB_SIZE, true)
    private val act = Tanh()

    override fun forward(input: ITensor): ITensor {
        val embeddingsForSequence = embedding[input] // (batch_size, seq_len, embedding_size)

        val batchSize = embeddingsForSequence.shape[0]
        val seqLen = embeddingsForSequence.shape[1]
        var hprev = start.broadcast(batchSize, -1) // (batch_size, hidden_size)

        // Iterate over sequence
        val hiddenStatesList = mutableListOf<ITensor>()
        for (i in 0 until seqLen) {
            val embeddingForItem =
                embeddingsForSequence[LongRange.ALL, i] // (batch_size, embedding_size)
            val xh = hprev.concat(embeddingForItem, 1)
            hprev = act(rnnCell(xh)) // (batch_size, hidden_size)
            hiddenStatesList.add(hprev)
        }

        // stack hidden states
        val hiddenStates = sciCore.stack(1, hiddenStatesList) // (batch_size, seq_len, hidden_size)
        return lmHead(hiddenStates)
    }

    override fun subModules(): List<IModule> {
        return listOf(rnnCell, lmHead)
    }

    override fun parameters(): List<ITensor> {
        return listOf(embedding, start, *rnnCell.parameters().toTypedArray(), *lmHead.parameters().toTypedArray())
    }

}