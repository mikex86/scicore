package me.mikex86.scicore.tests.makemore.gpt

import java.nio.file.Path
import java.util.*


val tokenizer = GPTTokenizer.getEncoder(Path.of("encoder.json"), Path.of("vocab.bpe"))

fun main() {
    val scanner = Scanner(System.`in`)
    while (true) {
        val input = scanner.nextLine()
        val tokens = tokenizer.encode(input)
        println("tokens: $tokens, decoded: ${tokenizer.decode(tokens)}")
    }
}