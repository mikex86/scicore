package me.mikex86.scicore.tests.makemore.gpt

import me.mikex86.matplotlib.jplot.JPlot
import me.mikex86.scicore.ISciCore
import me.mikex86.scicore.SciCore
import me.mikex86.scicore.data.DatasetIterator
import me.mikex86.scicore.graph.scopedRecording
import me.mikex86.scicore.nn.optim.Adam
import me.mikex86.scicore.nn.optim.Sgd
import me.mikex86.scicore.tests.makemore.gpt.data.TokenizedBinaryTokenStreamer
import me.mikex86.scicore.utils.use
import me.tongfei.progressbar.ProgressBarBuilder
import me.tongfei.progressbar.ProgressBarStyle
import java.awt.Color
import java.nio.file.Path
import java.util.*
import java.util.concurrent.CompletableFuture
import kotlin.math.log10

private const val BATCH_SIZE = 1
private const val N_TRAINING_STEPS = 500L

fun main() {
    val sciCore = SciCore()
    sciCore.seed(123)
    sciCore.setBackend(ISciCore.BackendType.CPU)

    // tiny gpt
    val config = GPTConfig(
        vocabSize = 50257,
        nLayers = 2,
        nHeads = 4,
        nEmbed = 32,
        blockSize = 256,
    )

    val model = GPTModel(sciCore, config)
    val optimizer = Adam(sciCore, 2.5e-3f, 0.9f, 0.95f, model.parameters())

    val datasetSupplier = TokenizedBinaryTokenStreamer(sciCore, Path.of("openwebtext.bin"), config.blockSize)

    val trainIt = DatasetIterator(BATCH_SIZE, datasetSupplier)

    val losses = FloatArray(N_TRAINING_STEPS.toInt())

    val jplot = JPlot()
    jplot.setName("TinyGPT2 Loss")

    val checkPointInterval = 50
    var lastStepStart = System.currentTimeMillis()

    sciCore.train()

    for (step in 0 until N_TRAINING_STEPS) {
        sciCore.backend.operationRecorder.scopedRecording {
            val batch = trainIt.next()
            batch.use { x, y ->
                val lossValue = model(x)
                    .let { logits -> sciCore.crossEntropy(logits.view(-1, logits.shape.last()), y.view(-1)) }
                    .use { loss ->
                        optimizer.step(loss)
                        loss.elementAsDouble()
                    }
                losses[step.toInt()] = lossValue.toFloat()
                val stepEnd = System.currentTimeMillis()
                println("Step ${step + 1}, Loss: $lossValue, Time: ${stepEnd - lastStepStart}ms")
                lastStepStart = stepEnd

                if (step % checkPointInterval == 0L) {
                    println("Saving checkpoint at step $step...")
                    model.save(Path.of("ckpts/gpt2-$step.scm"))
                    val lossesTillNow = losses.sliceArray(0..step.toInt())
                    Path.of("ckpts/gpt2-losses-$step.json").toFile().writeText(lossesTillNow.contentToString())
                    CompletableFuture.supplyAsync {
                        synchronized(jplot) {
                            jplot.clear()
                            jplot.plot(lossesTillNow, Color(46, 204, 113), true)
                            jplot.save(Path.of("ckpts/gpt2-lossplot-$step.png"))
                            jplot.show(false)
                            println("Checkpoint for step $step saved.")
                        }
                    }
                }
            }
        }
    }
    println("Finished training")
    model.save(Path.of("ckpts/tiny-gpt2.scm"))
}