package me.mikex86.scicore.tests.mnist

import me.mikex86.scicore.ISciCore
import me.mikex86.scicore.SciCore
import me.mikex86.scicore.data.DatasetIterator
import me.mikex86.scicore.graph.scopedRecording
import me.mikex86.scicore.tensor.toGrayScaleImage
import java.awt.Color
import java.awt.Graphics2D
import java.awt.RenderingHints
import java.nio.file.Path
import javax.swing.JFrame

fun main() {
    val sciCore = SciCore()
    sciCore.addBackend(ISciCore.BackendType.CPU)

    val net = MnistNet(sciCore)
    net.load(Path.of("mnist.scm"))

    val testIt = DatasetIterator(1, MnistDataSupplier(sciCore, train = false, shuffle = false))

    val jFrame = JFrame()
    jFrame.setSize(400, 400)
    jFrame.isVisible = true
    jFrame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE

    val g = jFrame.graphics as Graphics2D
    g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON)

    for (batch in testIt) {
        sciCore.backend.operationRecorder.scopedRecording {
            val input = batch.first
            val label = batch.second.argmax(1)

            val output = net.forward(input)
            val prediction = output.argmax(1)

            val bufferedImage = input.view(28, 28).transpose().toGrayScaleImage()
            jFrame.graphics.drawImage(bufferedImage, 0, 0, jFrame.width, jFrame.height, null)

            g.color = Color.WHITE
            g.drawString("Label: ${label.elementAsInt()}", 10, 50)
            g.drawString("Prediction: ${prediction.elementAsInt()}", 10, 65)

            Thread.sleep(1000)
        }
    }

}