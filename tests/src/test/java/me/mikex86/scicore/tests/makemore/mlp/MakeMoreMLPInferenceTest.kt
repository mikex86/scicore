package me.mikex86.scicore.tests.makemore.mlp

import me.mikex86.scicore.ISciCore
import me.mikex86.scicore.SciCore
import me.mikex86.scicore.tests.makemore.NamesCharacterMapping
import me.mikex86.scicore.tests.makemore.SlidingWindowEncoder
import java.util.Random
import kotlin.io.path.Path

private const val BLOCK_SIZE = 3

fun main() {
    val sciCore = SciCore()
    sciCore.addBackend(ISciCore.BackendType.CPU)

    val net = MakeMoreMLPNet(sciCore)
    net.load(Path("makemore.scm"))

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