package me.mikex86.scicore.tests.makemore

import java.lang.IllegalArgumentException

class SlidingWindowEncoder(private val blockSize: Int) {

    fun getWindow(str: String): ByteArray {
        if (str.length > blockSize) {
            throw IllegalArgumentException("String length of \"$str\" exceeds window size: $blockSize")
        }
        val window = ByteArray(blockSize) { NamesCharacterMapping.charToIndex['.']!!.toByte() }
        var idx = 0
        for (c in str.reversed()) {
            window[window.size - idx - 1] = NamesCharacterMapping.charToIndex[c]
                ?: throw IllegalArgumentException("Cannot encode character \'$c\' with model character mapping")
            idx++
        }
        return window
    }

}