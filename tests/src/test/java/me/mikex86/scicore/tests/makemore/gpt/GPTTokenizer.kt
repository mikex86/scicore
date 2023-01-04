package me.mikex86.scicore.tests.makemore.gpt

import com.google.gson.GsonBuilder
import com.google.gson.JsonObject
import java.nio.file.Path
import java.util.WeakHashMap

class GPTTokenizer private constructor(
    private val encoder: Map<String, Int>,
    bpeMerges: List<Pair<String, String>>
) {

    private val decoder = encoder.entries.associate { it.value to it.key }
    private val bpeRanks = bpeMerges.withIndex().associate { (i, pair) -> pair to i }
    private val pattern = Regex("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+")
    private val cache = WeakHashMap<String, String>()

    private val byteEncoder: Map<Byte, String> = run {
        val bs = ((33..126).toList() + (161..172).toList() + (174..255).toList()).toMutableList()
        val cs = bs.toMutableList()
        var n = 0
        for (b in 0..255) {
            if (b !in bs) {
                bs.add(b)
                cs.add(256 + n)
                n += 1
            }
        }
        bs.map { it.toUByte().toByte() }.zip(cs.map { it.toChar().toString() }).toMap()
    }
    private val byteDecoder = byteEncoder.entries.associate { it.value to it.key }

    companion object {
        fun getEncoder(encoderJson: Path, vocabBpe: Path): GPTTokenizer {
            val gson = GsonBuilder().create()
            val tokenMapping = gson
                .fromJson(encoderJson.toFile().readText(), JsonObject::class.java)
                .entrySet()
                .associate { it.key to it.value.asInt }
            val bpeMerges = vocabBpe.toFile()
                .readLines()
                .drop(1)
                .map { it.split(" ") }
                .map { it[0] to it[1] }
            return GPTTokenizer(tokenMapping, bpeMerges)
        }
    }

    private fun getPairs(chars: List<String>): Set<Pair<String, String>> {
        val pairs = mutableSetOf<Pair<String, String>>()
        var prevChar = chars[0]
        for (char in chars.drop(1)) {
            pairs.add(prevChar to char)
            prevChar = char
        }
        return pairs
    }

    private fun bpe(token: String): String {
        cache[token]?.let { return it }

        var word = token.toCharArray().map { it.toString() }
        var pairs = getPairs(word)

        if (pairs.isEmpty()) {
            return token
        }

        while (true) {
            val bigram = run {
                var minBigram = pairs.first()
                var minRank = bpeRanks[minBigram] ?: Int.MAX_VALUE
                for (pair in pairs) {
                    val rank = bpeRanks[pair] ?: Int.MAX_VALUE
                    if (rank < minRank) {
                        minBigram = pair
                        minRank = rank
                    }
                }
                if (minRank == Int.MAX_VALUE) {
                    return@run null
                }
                minBigram
            }
            val (first, second) = bigram ?: break
            val newWord = mutableListOf<String>()
            var i = 0
            while (i < word.size) {
                val j = with(word) {
                    for (j in i..lastIndex) {
                        if (this[j] == first) {
                            return@with j
                        }
                    }
                    return@with null
                }
                if (j == null) {
                    newWord.addAll(word.subList(i, word.size))
                    break
                }
                newWord.addAll(word.subList(i, j))
                i = j

                i += if (word[i] == first && i < word.size - 1 && word[i + 1] == second) {
                    newWord.add(first + second)
                    2
                } else {
                    newWord.add(word[i])
                    1
                }
            }
            word = newWord
            if (word.size == 1) {
                break
            }
            pairs = getPairs(word)
        }

        val result = word.joinToString(" ")
        cache[token] = result
        return result
    }

    fun encode(text: String): List<Int> {
        val bpeTokens = mutableListOf<Int>()
        val tokens = pattern.findAll(text).map { it.value }.toList()
        for (token in tokens) {
            bpeTokens.addAll(
                token.toByteArray(Charsets.UTF_8).joinToString("") { byteEncoder[it]!! }
                    .let { bytesAsAsciiChars -> bpe(bytesAsAsciiChars).split(" ") }
                    .map { bpeToken -> encoder[bpeToken] ?: throw IllegalArgumentException("Unknown token: $bpeToken") }
            )
        }
        return bpeTokens
    }

    fun decode(tokens: List<Int>): String {
        val text = tokens.joinToString("") { decoder[it] ?: throw IllegalArgumentException("Unknown token: $it") }
        return text.toCharArray()
            .map { byteDecoder[it.toString()]!! }
            .toByteArray()
            .decodeToString()
    }
}
