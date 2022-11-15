package me.mikex86.scicore.tests.makemore

import me.mikex86.scicore.SciCore
import me.mikex86.scicore.tensor.ITensor
import me.mikex86.scicore.utils.ShapeUtils
import java.util.function.Supplier
import kotlin.io.path.Path
import kotlin.io.path.readLines
import kotlin.random.Random

class NamesDatasetSupplier(
    private val sciCore: SciCore,
    private val blockSize: Int,
    training: Boolean
) :
    Supplier<Pair<ITensor, ITensor>> {

    private val lines = Path("names.txt").readLines(Charsets.UTF_8)

    private val chars: List<Char> = lines.flatMap { it.toCharArray().asIterable() }.distinct().sorted()

    private val charToIndex: Map<Char, Byte> = chars.withIndex().associate { it.value to (it.index + 1).toByte() }
        .toMutableMap()
        .apply {
            put('.', 0) // '.' character is used as padding
        }

    private val indexToChar: Map<Byte, Char> = charToIndex.entries.associate { it.value to it.key }

    private fun buildDataset(lines: List<String>): Pair<ITensor, ITensor> {
        val x = mutableListOf<ByteArray>()
        val y = mutableListOf<Byte>()
        for (line in lines) {
            var context = ByteArray(blockSize) { charToIndex['.']!! } // init with padding special char
            for (ch in "$line.") {
                val ix = charToIndex[ch]!!
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

    private val random = Random(123)

    private val x: ITensor
    private val y: ITensor

    init {
        lines.shuffled(random)

        val n = (lines.size * 0.8).toInt()
        val (x, y) = buildDataset(if (training) lines.subList(0, n) else lines.subList(n, lines.size))
        this.x = x
        this.y = y
    }


    override fun get(): Pair<ITensor, ITensor> {
        return Pair(
            x.getView(random.nextLong(x.shape[0])),
            y.getView(random.nextLong(y.shape[0]))
        )
    }

}