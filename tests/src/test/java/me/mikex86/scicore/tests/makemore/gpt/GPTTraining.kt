package me.mikex86.scicore.tests.makemore.gpt

import com.google.gson.Gson
import com.google.gson.JsonArray
import me.mikex86.matplotlib.jplot.JPlot
import me.mikex86.scicore.ISciCore
import me.mikex86.scicore.SciCore
import me.mikex86.scicore.data.DatasetIterator
import me.mikex86.scicore.graph.scopedRecording
import me.mikex86.scicore.nn.optim.Adam
import me.mikex86.scicore.tensor.DataType
import me.mikex86.scicore.tensor.get
import me.mikex86.scicore.tests.makemore.gpt.data.TokenizedBinaryTokenStreamer
import me.mikex86.scicore.utils.use
import java.awt.Color
import java.io.File
import java.io.FileReader
import java.nio.file.Path
import java.util.concurrent.CompletableFuture

private const val BATCH_SIZE = 1
private const val N_TRAINING_STEPS = 50000L

fun main() {
    val sciCore = SciCore()
    sciCore.seed(123)
    sciCore.addBackend(ISciCore.BackendType.CPU)
//    sciCore.addBackend(ISciCore.BackendType.CUDA)

    // tiny gpt
    val config = GPTConfig(
        vocabSize = 50257,
        nLayers = 2,
        nHeads = 8,
        nEmbed = 1024,
        blockSize = 256,
    )

    val model = GPTModel(sciCore, config)

    val losses = sciCore.zeros(DataType.FLOAT32, N_TRAINING_STEPS)

    val lastStep = File("ckpts/").listFiles()
        ?.filter { it.name.contains(".scm") }?.toList()
        ?.maxOfOrNull { it.name.substringAfter("gpt2-").substringBefore(".scm").toLong() } ?: 0L

    if (lastStep != 0L) {
        println("Loading checkpoint $lastStep")
        model.load(Path.of("ckpts/gpt2-$lastStep.scm")) // load checkpoint
        Gson().fromJson(FileReader("ckpts/gpt2-losses-$lastStep.json"), JsonArray::class.java).let { lossesArray ->
            for (i in 0 until lastStep) {
                losses.setFloat(lossesArray[i.toInt()].asFloat, i)
            }
        }
    }

    val optimizer = Adam(sciCore, 2.5e-5f, 0.9f, 0.95f, model.parameters())

    val datasetSupplier = TokenizedBinaryTokenStreamer(sciCore, Path.of("openwebtext.bin"), config.blockSize)

    val trainIt = DatasetIterator(BATCH_SIZE, datasetSupplier)

    val jplot = JPlot()
    jplot.setName("TinyGPT2 Loss")

    val checkPointInterval = 100
    var lastStepStart = System.currentTimeMillis()

    sciCore.train()

    for (step in lastStep until N_TRAINING_STEPS) {
        sciCore.backend.operationRecorder.scopedRecording {
            val batch = trainIt.next()
            batch.use { x, y ->
                val lossValue = model(x)
                    .use { logits -> sciCore.crossEntropy(logits.view(-1, logits.shape.last()), y.view(-1)) }
                    .use { loss ->
                        optimizer.step(loss)
                        loss.elementAsDouble()
                    }
                losses.setFloat(lossValue.toFloat(), step)
                val stepEnd = System.currentTimeMillis()
                println("Step ${step + 1}, Loss: $lossValue, Time: ${stepEnd - lastStepStart}ms")
                lastStepStart = stepEnd

                if (step % checkPointInterval == 0L) {
                    println("Saving checkpoint at step $step...")
                    model.save(Path.of("ckpts/gpt2-$step.scm"))
                    val lossesTillNow = losses[0 until step]
                    val lossesTillNowMean = lossesTillNow.view(-1, 10).mean(1, false)
                    val lossesTillNowArray = FloatArray(lossesTillNow.shape.first().toInt()) { i ->
                        lossesTillNow.getFloat(i.toLong())
                    }
                    val lossesTillNowMeanArray =
                        FloatArray(lossesTillNowMean.shape.first().toInt()) {
                            lossesTillNowMean.getFloat(it.toLong())
                        }
                    Path.of("ckpts/gpt2-losses-$step.json").toFile().writeText(lossesTillNowArray.contentToString())
                    CompletableFuture.supplyAsync {
                        synchronized(jplot) {
                            jplot.clear()
                            jplot.plot(lossesTillNowMeanArray, Color(46, 204, 113), true)
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