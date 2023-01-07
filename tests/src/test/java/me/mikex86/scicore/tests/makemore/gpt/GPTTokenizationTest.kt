package me.mikex86.scicore.tests.makemore.gpt

import java.nio.file.Path
import java.util.*

fun main() {
    val tokenizer = GPTTokenizer.getEncoder(
        Path.of("encoder.json"),
        Path.of("vocab.bpe"),
    )

    val scanner = Scanner(System.`in`)
    while (true) {
        print("Enter text: ")
        val text = scanner.nextLine()
        val tokens = tokenizer.encode(text)
        println("Tokens: $tokens")
        println("Text: ${tokenizer.decode(tokens)}")
    }
}