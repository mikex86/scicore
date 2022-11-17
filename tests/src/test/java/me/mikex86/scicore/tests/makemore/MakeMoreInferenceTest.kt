package me.mikex86.scicore.tests.makemore

import me.mikex86.scicore.ISciCore
import me.mikex86.scicore.SciCore
import kotlin.io.path.Path
import kotlin.random.Random

private const val BLOCK_SIZE = 8

fun main() {
    val sciCore = SciCore()
    sciCore.setBackend(ISciCore.BackendType.CPU)

    val net = MakeMoreNet(sciCore)
    net.load(Path("makemore.scm"))

    val slidingWindowEncoder = SlidingWindowEncoder(blockSize = BLOCK_SIZE)
    var windowString = "emil"
    var totalString = "emil"

    val random = Random(312)
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
                for (idx in 0 until probs.shape[1]) {
                    probValues[idx.toInt()] = probs.getAsFloat(0, idx)
                }
                probValues.shuffle(random)
                var cumulativeProb = 0f
                var chosenIndex = 0
                val v = random.nextFloat()
                for ((idx, prob) in probValues.withIndex()) {
                    cumulativeProb += prob
                    if (cumulativeProb >= v) {
                        chosenIndex = idx
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