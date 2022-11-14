package me.mikex86.scicore.tensor

import java.awt.image.BufferedImage

fun ITensor.toGrayScaleImage(): BufferedImage {
    val width = shape[0]
    val height = shape[1]
    val bufferedImage = BufferedImage(width.toInt(), height.toInt(), BufferedImage.TYPE_INT_ARGB)
    for (y in 0 until height) {
        for (x in 0 until width) {
            val value = if (dataType.isFloatingPoint) { (getAsFloat(x, y) * 255).toInt() }  else { getAsInt(x, y) }
            val argb = (255 shl 24) or (value shl 16) or (value shl 8) or value
            bufferedImage.setRGB(x.toInt(), y.toInt(), argb)
        }
    }
    return bufferedImage
}