package me.mikex86.scicore.tests.makemore

import me.mikex86.scicore.SciCore
import me.mikex86.scicore.tensor.ITensor
import me.mikex86.scicore.utils.ShapeUtils
import java.util.Random
import java.util.function.Supplier

class NamesDatasetSupplier(
    private val sciCore: SciCore,
    private val blockSize: Int,
    training: Boolean,
    shuffle: Boolean
) :
    Supplier<Pair<ITensor, ITensor>> {


    private fun buildDataset(lines: List<String>): Pair<ITensor, ITensor> {
        val x = mutableListOf<ByteArray>()
        val y = mutableListOf<Byte>()
        for (line in lines) {
            var context =
                ByteArray(blockSize) { NamesCharacterMapping.charToIndex['.']!! } // init with padding special char
            for (ch in "$line.") {
                val ix = NamesCharacterMapping.charToIndex[ch]!!
                x.add(context)
                y.add(ix)
                context = context.sliceArray(1 until blockSize) + ix
            }
        }
        val xTensor = sciCore.matrix(x.toTypedArray())
        val yTensor = sciCore.array(y.toTypedArray().toByteArray())
        println("xTensor: ${ShapeUtils.toString(xTensor.shape)} yTensor: ${ShapeUtils.toString(yTensor.shape)}")
        return Pair(xTensor, yTensor)
    }

    private val random = if (shuffle) {
        Random(123)
    } else {
        null
    }

    val x: ITensor
    val y: ITensor

    init {
        val lines =
            random?.let { random -> NamesCharacterMapping.lines.shuffled(random) } ?: NamesCharacterMapping.lines

        val n = (lines.size * 0.8).toInt()
        val (x, y) = buildDataset(if (training) lines.subList(0, n) else lines.subList(n, lines.size))
        this.x = x
        this.y = y
    }

    private var idx = 0L
    override fun get(): Pair<ITensor, ITensor> {
        val idx = random?.nextInt(x.shape[0].toInt()) ?: idx++.toInt()
        return Pair(
            x.getView(idx.toLong()),
            y.getView(idx.toLong())
        )
    }

}