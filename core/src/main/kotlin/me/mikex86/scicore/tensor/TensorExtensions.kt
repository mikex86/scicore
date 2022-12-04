package me.mikex86.scicore.tensor

import me.mikex86.scicore.ranges.ALL
import java.awt.image.BufferedImage

fun ITensor.toGrayScaleImage(): BufferedImage {
    val width = shape[0]
    val height = shape[1]
    val bufferedImage = BufferedImage(width.toInt(), height.toInt(), BufferedImage.TYPE_INT_ARGB)
    for (y in 0 until height) {
        for (x in 0 until width) {
            val value = if (dataType.isFloatingPoint) {
                (getAsFloat(x, y) * 255).toInt()
            } else {
                getAsInt(x, y)
            }
            val argb = (255 shl 24) or (value shl 16) or (value shl 8) or value
            bufferedImage.setRGB(x.toInt(), y.toInt(), argb)
        }
    }
    return bufferedImage
}

operator fun ITensor.unaryMinus() = multiply(-1f)


operator fun ITensor.get(vararg indices: Any): ITensor {
    val backend = sciCoreBackend
    if (indices.size > shape.size) {
        throw IllegalArgumentException("Too many indices (expected ${shape.size} or fewer)")
    }
    return indices.withIndex().map { (dim, idx) ->
        when (idx) {
            is Int -> backend.createTensor(DataType.INT32, longArrayOf(1L)).apply {
                setInt(idx, 0)
            }

            is Long -> backend.createTensor(DataType.INT64, longArrayOf(1L)).apply {
                setLong(idx, 0)
            }

            is ITensor -> idx
            is IntRange -> idx.let { range ->
                if (range == IntRange.ALL) {
                    backend.createTensor(DataType.INT64, longArrayOf(shape[dim])).apply {
                        for (i in 0 until this@get.shape[dim]) {
                            setLong(i, i)
                        }
                    }
                } else {
                    backend.createTensor(DataType.INT32, longArrayOf(range.last - range.first + 1L)).apply {
                        for ((i, v) in (range.first..range.last).withIndex()) {
                            setInt(v, i.toLong())
                        }
                    }
                }
            }

            is LongRange -> idx.let { range ->
                if (range == LongRange.ALL) {
                    backend.createTensor(DataType.INT64, longArrayOf(shape[dim])).apply {
                        for (i in 0 until this@get.shape[dim]) {
                            setLong(i, i)
                        }
                    }
                } else {
                    backend.createTensor(DataType.INT64, longArrayOf(range.last - range.first + 1L)).apply {
                        for ((i, v) in (range.first..range.last).withIndex()) {
                            setLong(v, i.toLong())
                        }
                    }
                }
            }

            else -> throw IllegalArgumentException("Invalid index type: ${idx::class}")
        }
    }.let { indexTensors ->
        return@let this@get.get(*indexTensors.toTypedArray())
    }
}