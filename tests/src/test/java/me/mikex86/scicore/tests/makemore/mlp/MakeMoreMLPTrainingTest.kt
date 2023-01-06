package me.mikex86.scicore.tests.makemore.mlp

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
import me.mikex86.scicore.tensor.*
import me.mikex86.scicore.tests.makemore.NamesCharacterMapping
import me.mikex86.scicore.tests.makemore.NamesSlidingWindowDatasetSupplier
import me.mikex86.scicore.tests.makemore.SlidingWindowEncoder
import me.mikex86.scicore.utils.use
import me.tongfei.progressbar.ProgressBarBuilder
import me.tongfei.progressbar.ProgressBarStyle
import java.awt.Color
import java.nio.file.Path
import java.util.*

private const val BATCH_SIZE = 32
private const val BLOCK_SIZE = 3
private const val EMBEDDING_SIZE = 8
private const val N_HIDDEN = 200
private const val VOCAB_SIZE = 26 + 1 // 26 letters + 1 padding char

private const val N_TRAINING_STEPS = 200_000L
private const val INITIAL_LEARNING_RATE = 0.1f
private const val END_LEARNING_RATE = 0.05f

fun main() {
    val sciCore = SciCore()
    sciCore.setBackend(ISciCore.BackendType.CPU)
    sciCore.seed(123)

    val namesSupplier = NamesSlidingWindowDatasetSupplier(sciCore, BLOCK_SIZE, training = true, shuffle = true)
    val trainIt = DatasetIterator(BATCH_SIZE, namesSupplier)

    val net = MakeMoreMLPNet(sciCore)

    val optimizer = Sgd(sciCore, INITIAL_LEARNING_RATE, END_LEARNING_RATE, N_TRAINING_STEPS, net.parameters())
    var lossValue = -1.0

    // loss on dataset
    sciCore.backend.operationRecorder.scopedRecording {
        val x = namesSupplier.x
        val y = namesSupplier.y
        val loss = net(x)
            .use { logits -> sciCore.crossEntropy(logits, y) }
            .use { loss -> loss.elementAsDouble() }
        println("Loss on dataset before training: $loss")
    }

    println("Start training...")
    val start = System.currentTimeMillis()
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
                            .use { logits -> sciCore.crossEntropy(logits, y) }
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
    net.save(Path.of("makemore.scm"))

    val jplot = JPlot()
    jplot.setName("MLP Language Model Training")
    jplot.setXLabel("Step")
    jplot.setYLabel("Loss")
    val avgLosses = losses.view(-1, 100).mean(1, false)
    val lossesArray = FloatArray(avgLosses.numberOfElements.toInt()) { avgLosses.getFloat(it.toLong()) }
    jplot.plot(lossesArray, Color(46, 204, 113), true)
    jplot.save(Path.of("mlplm_loss.png"))


    // loss on dataset
    sciCore.backend.operationRecorder.scopedRecording {
        val x = namesSupplier.x
        val y = namesSupplier.y
        val loss = net(x)
            .use { logits -> sciCore.crossEntropy(logits, y) }
            .use { loss -> loss.elementAsDouble() }
        println("Loss on dataset: $loss")
    }

    // Print number of executed operations
    println("Number of performed operations: " + GraphExecutor.getNumOperations())

    // Inference
    run {
        val slidingWindowEncoder = SlidingWindowEncoder(blockSize = BLOCK_SIZE)
        val random = Random(123)
        for (i in 0 until 20) {
            var windowString = ""
            var totalString = ""

            while (!windowString.endsWith(".")) {
                val window = slidingWindowEncoder.getWindow(windowString)
                val windowTensor = sciCore.array(window)
                val nextChar = net(windowTensor)
                    .use { logits ->
                        sciCore.softmax(logits, 1)
                    }
                    .use { probs ->
                        // multinomial distribution sampling
                        val probValues = FloatArray(probs.shape[1].toInt())
                        val probIndices = mutableListOf<Byte>()
                        for (idx in 0 until probs.shape[1]) {
                            probValues[idx.toInt()] = probs.getAsFloat(0, idx)
                            probIndices.add(idx.toByte())
                        }
                        probIndices.shuffle(random)
                        var cumulativeProb = 0f
                        var chosenIndex = 0
                        val u = random.nextFloat()
                        for (idx in probIndices) {
                            val prob = probValues[idx.toInt()]
                            cumulativeProb += prob
                            if (cumulativeProb >= u) {
                                chosenIndex = idx.toInt()
                                break
                            }
                        }
                        chosenIndex
                    }.let { idx ->
                        NamesCharacterMapping.indexToChar[idx.toByte()]!!
                    }
                totalString += nextChar
                windowString += nextChar
                if (windowString.length > BLOCK_SIZE) {
                    windowString = windowString.substring(1)
                }
            }
            println(totalString)
        }
    }
}

class MakeMoreMLPNet(sciCore: SciCore) : IModule {

    private val embedding = sciCore.gaussian(
        DataType.FLOAT32,
        VOCAB_SIZE.toLong(),
        EMBEDDING_SIZE.toLong()
    )

    private val fc1 = Linear(sciCore, DataType.FLOAT32, (EMBEDDING_SIZE * BLOCK_SIZE).toLong(), N_HIDDEN.toLong(), true)
    private val fc2 = Linear(sciCore, DataType.FLOAT32, N_HIDDEN.toLong(), VOCAB_SIZE.toLong(), true)
    private val act = Tanh()

    override fun forward(input: ITensor): ITensor {
        // input: (batchSize, blockSize)
        val logits = embedding[input] // embeddingsForSequence: (batchSize, blockSize, embeddingSize)
            .use { embeddingsForSequence ->
                // embeddingsForSequenceFlat: (batchSize, blockSize * embeddingSize)
                embeddingsForSequence.view(
                    -1,
                    (EMBEDDING_SIZE * BLOCK_SIZE).toLong()
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