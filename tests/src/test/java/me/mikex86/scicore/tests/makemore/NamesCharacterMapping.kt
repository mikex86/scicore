package me.mikex86.scicore.tests.makemore

import kotlin.io.path.Path
import kotlin.io.path.readLines

object NamesCharacterMapping {

    val lines = Path("names.txt").readLines(Charsets.UTF_8)

    private val chars: List<Char> = lines.flatMap { it.toCharArray().asIterable() }.distinct().sorted()

    val charToIndex: Map<Char, Byte> = chars.withIndex().associate { it.value to (it.index + 1).toByte() }
        .toMutableMap()
        .apply {
            put('.', 0) // '.' character is used as padding
        }

    val indexToChar: Map<Byte, Char> = charToIndex.entries.associate { it.value to it.key }


}