package me.mikex86.scicore.tests.makemore.bigram

import me.mikex86.scicore.SciCore
import me.mikex86.scicore.tensor.DataType
import me.mikex86.scicore.tensor.ITensor
import me.mikex86.scicore.tests.makemore.NamesCharacterMapping

val sciCore = SciCore()

fun tokenize(text: String): List<Int> {
    return text.toCharArray().map { NamesCharacterMapping.charToIndex[it]!!.toInt() }.toList()
}

fun bigramLanguageModel(text: String): ITensor {
    val tokens = tokenize(text)
    val P = sciCore.zeros(
        DataType.FLOAT32,
        NamesCharacterMapping.charToIndex.size.toLong(),
        NamesCharacterMapping.charToIndex.size.toLong()
    )
    for (i in 0 until tokens.size - 1) {
        val token = tokens[i]
        val newToken = tokens[i + 1]
        val row = P.getView(token.toLong())
        row.setFloat(row.getFloat(newToken.toLong()) + 1, newToken.toLong()) // += 1
    }
    return P / P.reduceSum(1, true)
}

fun sampleFromBigram(prevToken: Int, P: ITensor): Int {
    return sciCore.multinomial(P.getView(prevToken.toLong()), 1).getAsInt(0)
}

val P = bigramLanguageModel(
    NamesCharacterMapping.lines.joinToString(
        // special start and end token
        separator = ".",
        prefix = ".",
        postfix = "."
    )
)

fun generateText(): String {
    val name = StringBuilder()
    var prevToken = 0 // start token
    while (true) {
        val token = sampleFromBigram(prevToken, P)
        if (token == 0) { // end token
            break
        }
        name.append(NamesCharacterMapping.indexToChar[token.toByte()])
        prevToken = token
    }
    return name.toString()
}

fun main() {
    sciCore.seed(12345)
    repeat(10) {
        println(generateText())
    }
}