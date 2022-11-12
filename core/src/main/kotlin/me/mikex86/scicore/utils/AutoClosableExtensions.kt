package me.mikex86.scicore.utils

fun <F : AutoCloseable, S : AutoCloseable> Pair<F, S>.use(block: (F, S) -> Unit) {
    first.use { first ->
        second.use { second ->
            block(first, second)
        }
    }
}