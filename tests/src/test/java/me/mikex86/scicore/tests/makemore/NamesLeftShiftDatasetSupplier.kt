package me.mikex86.scicore.tests.makemore

import me.mikex86.scicore.SciCore
import me.mikex86.scicore.tensor.ITensor
import java.util.*
import java.util.function.Supplier

class NamesLeftShiftDatasetSupplier(
    private val sciCore: SciCore,
    training: Boolean,
    shuffle: Boolean
) : Supplier<Pair<ITensor, ITensor>> {

    private val maxWordLength: Int

    private val xs: List<ITensor>
    private val ys: List<ITensor>

    private val random = if (shuffle) {
        Random(123)
    } else {
        null
    }

    init {
        val lines = NamesCharacterMapping.lines
        maxWordLength = lines.maxOf { it.length }

        val n = (lines.size * 0.8).toInt()
        val (xs, ys) = buildDataset(if (training) lines.subList(0, n) else lines.subList(n, lines.size))
        this.xs = xs
        this.ys = ys
    }

    private fun buildDataset(words: List<String>): Pair<List<ITensor>, List<ITensor>> {
        val xs = mutableListOf<ITensor>()
        val ys = mutableListOf<ITensor>()
        for (w in words) {
            val x = ByteArray(maxWordLength + 2) { 0 }
            for (i in 1..w.length) {
                x[i] = NamesCharacterMapping.charToIndex[w[i - 1]]!!
            }
            xs.add(sciCore.array(x))

            var y = x.sliceArray(1..w.length + 1)
            y += ByteArray(maxWordLength + 2 - y.size) { -1 }
            ys.add(sciCore.array(y))
        }
        return Pair(xs, ys)
    }

    private var idx = 0

    override fun get(): Pair<ITensor, ITensor> {
        val idx = random?.nextInt(xs.size) ?: idx++
        return Pair(xs[idx], ys[idx])
    }

}