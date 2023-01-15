package me.mikex86.scicore.tests.makemore.gpt

import me.mikex86.scicore.ISciCore
import me.mikex86.scicore.SciCore
import me.mikex86.scicore.graph.scopedRecording
import me.mikex86.scicore.ranges.ALL
import me.mikex86.scicore.tensor.get
import java.nio.file.Path

fun generateText(
    sciCore: SciCore,
    prompt: String,
    numTokens: Int,
    model: GPTModel,
    tokenizer: GPTTokenizer,
    top1: Boolean = false,
    temperature: Float = 1.0f
): String {
    val promptTokens = tokenizer.encode(prompt).toMutableList()
    val outputStringBuilder = StringBuilder()

    for (i in 0 until numTokens) {
        val input = sciCore.array(promptTokens.toIntArray())
            .view(1, -1)
        val seqLen = input.shape.last()
        sciCore.backend.operationRecorder.scopedRecording {
            model(input)
                .let { logits ->
                    logits / temperature
                }
                .let { logits ->
                    logits.softmax(2)
                }
                .use { blockLogits -> blockLogits[LongRange.ALL, seqLen - 1, LongRange.ALL] }
                .use { probs ->
                    if (!top1) {
                        sciCore.multinomial(
                            probs.view(1, -1),
                            1
                        )
                    } else {
                        probs.argmax(0)
                    }
                }
                .use { nextToken ->
                    nextToken.getAsInt(0)
                }
                .let { nextToken ->
                    val text = tokenizer.decode(listOf(nextToken))
                    outputStringBuilder.append(text)
                    promptTokens.add(nextToken)
                }
        }
    }
    return outputStringBuilder.toString()
}

fun main() {
    val sciCore = SciCore()
    sciCore.seed(123)
    sciCore.addBackend(ISciCore.BackendType.CPU)


    val config = GPTConfig(
        vocabSize = 50257,
        nLayers = 2,
        nHeads = 4,
        nEmbed = 256,
        blockSize = 256,
    )

    val tokenizer = GPTTokenizer.getEncoder(
        Path.of("encoder.json"),
        Path.of("vocab.bpe"),
    )

    val model = GPTModel(sciCore, config)
    model.load(Path.of("ckpts/gpt2-12300.scm"))

    val prompt = "What is the meaning of life? The meaning of life is "
    val numTokens = 20
    val output = generateText(sciCore, prompt, numTokens, model, tokenizer, false, 0.5f)
    println("Prompt: $prompt")
    println("Completion: $output")
}