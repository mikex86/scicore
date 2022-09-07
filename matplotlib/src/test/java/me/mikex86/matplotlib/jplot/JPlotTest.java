package me.mikex86.matplotlib.jplot;

import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

class JPlotTest {

    @Test
    void plotSinAndCos() {
        JPlot jPlot = new JPlot();
        float[] data = new float[250];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) Math.sin(i / 25f);
        }
        float[] data2 = new float[250];
        for (int i = 0; i < data2.length; i++) {
            data2[i] = (float) Math.cos(i / 25f);
        }
        jPlot.setName("Sin and Cos");
        jPlot.setXLabel("time");
        jPlot.setYLabel("translation");
        jPlot.plot(data, new Color(46, 204, 113), true);
        jPlot.plot(data2, new Color(52, 152, 219), false);
        jPlot.show(true);
    }

    @Test
    void plotLog() {
        JPlot jPlot = new JPlot();
        float[] data = new float[50];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) Math.log(i + 1);
        }
        jPlot.setName("Log");
        jPlot.setXLabel("time");
        jPlot.setYLabel("log");
        jPlot.plot(data, new Color(46, 204, 113), true);
        jPlot.show(true);
    }

    @Test
    void plotParabola() {
        JPlot jPlot = new JPlot();
        float[] data = new float[50];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) Math.pow(-25 + i, 2);
        }
        jPlot.setName("Parabola");
        jPlot.setXLabel("time");
        jPlot.setYLabel("translation");
        jPlot.plot(data, new Color(46, 138, 204), true);
        jPlot.show(true);
    }
}