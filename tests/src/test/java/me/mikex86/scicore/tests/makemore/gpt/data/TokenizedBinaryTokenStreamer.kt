package me.mikex86.scicore.tests.makemore.gpt.data

import me.mikex86.scicore.SciCore
import me.mikex86.scicore.tensor.ITensor
import java.io.RandomAccessFile
import java.nio.file.Path
import java.util.function.Supplier
import kotlin.random.Random

class TokenizedBinaryTokenStreamer(private val sciCore: SciCore, tokenFile: Path, private val blockSize: Int) :
    Supplier<Pair<ITensor, ITensor>> {

    private val trainBin = RandomAccessFile(tokenFile.toFile(), "r")

//    private val random = Random(123)
    private var idx = 0L

    override fun get(): Pair<ITensor, ITensor> {
//        val idx = random.nextLong(trainBin.length() / Short.SIZE_BYTES)
        trainBin.seek(idx * Short.SIZE_BYTES)
        idx += blockSize
        val tokens = IntArray(blockSize + 1)
        val bytes = ByteArray(Short.SIZE_BYTES * (blockSize + 1))
        trainBin.read(bytes)
        for (i in 0 until blockSize + 1) {
            // read little endian unsigned shorts
            tokens[i] = (bytes[i * Short.SIZE_BYTES].toInt() and 0xFF) or
                    ((bytes[i * Short.SIZE_BYTES + 1].toInt() and 0xFF) shl 8)
        }
        val x = sciCore.array(tokens.sliceArray(0 until blockSize))
        val y = sciCore.array(tokens.sliceArray(1 until tokens.size))
        return x to y

    }

}