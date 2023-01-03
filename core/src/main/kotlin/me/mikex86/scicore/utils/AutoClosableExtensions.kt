package me.mikex86.scicore.utils

fun <R, F : AutoCloseable, S : AutoCloseable> Pair<F, S>.use(block: (F, S) -> R): R {
    first.use { first ->
        second.use { second ->
            return block(first, second)
        }
    }
}

fun <R, F : AutoCloseable, S : AutoCloseable, T : AutoCloseable> Triple<F, S, T>.use(block: (F, S, T) -> R): R {
    first.use { first ->
        second.use { second ->
            third.use { third ->
                return block(first, second, third)
            }
        }
    }
}

fun <R, T : AutoCloseable> List<T>.use(block: (List<T>) -> R): R {
    val list = mutableListOf<T>()
    try {
        for (item in this) {
            item.use { list.add(it) }
        }
        return block(list)
    } finally {
        for (item in list) {
            item.close()
        }
    }
}