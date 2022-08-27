package me.mikex86.matplotlib.jplot;

import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

class JPlotTest {

    @Test
    void plot() {
        JPlot jPlot = new JPlot();
        float[] data = new float[50];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) Math.sin(i / 5.0f);
        }
        float[] data2 = new float[50];
        for (int i = 0; i < data2.length; i++) {
            data2[i] = (float) Math.cos(i / 5.0f);
        }
        jPlot.setXLabel("time");
        jPlot.setYLabel("translation");
        jPlot.plot(data, new Color(46, 204, 113));
        jPlot.plot(data2, new Color(52, 152, 219));
        BufferedImage image = jPlot.render();
        try {
            ImageIO.write(image, "png", new java.io.File("plot.png"));
        } catch (IOException e) {
            fail(e);
        }
    }
}