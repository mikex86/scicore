package me.mikex86.matplotlib.jplot;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.jetbrains.skija.*;
import org.jetbrains.skija.Canvas;
import org.jetbrains.skija.Font;
import org.jetbrains.skija.Image;
import org.jetbrains.skija.Paint;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class JPlot {

    @Nullable
    private String xLabel;

    @Nullable
    private String yLabel;

    @Nullable
    private Float beginX = null, endX = null;

    @Nullable
    private Float beginY = null, endY = null;

    private static final int SCALE_FACTOR = 2;

    private static final int DEFAULT_WIDTH = 800, DEFAULT_HEIGHT = 600;

    private int width = DEFAULT_WIDTH * SCALE_FACTOR, height = DEFAULT_HEIGHT * SCALE_FACTOR;

    private static final int DEFAULT_MARGIN = 80 * SCALE_FACTOR;

    private static final float DEFAULT_STROKE_WIDTH = 4.0f * SCALE_FACTOR;

    private static final int DEFAULT_LABEL_FONT_SIZE = 24 * SCALE_FACTOR;

    private static final int DEFAULT_NAME_TEXT_FONT_SIZE = 28 * SCALE_FACTOR;

    @NotNull
    private final List<Series> series = new ArrayList<>();

    @Nullable
    private String name = null;

    public void setName(@NotNull String name) {
        this.name = name;
    }

    private record Series(float @NotNull [] data, int color, boolean fill) {
    }

    public void setXLabel(@NotNull String label) {
        this.xLabel = label;
    }

    public void setYLabel(@NotNull String label) {
        this.yLabel = label;
    }

    public void setXInterval(float begin, float endInclusive) {
        this.beginX = begin;
        this.endX = endInclusive;
    }

    public void setYInterval(float begin, float endInclusive) {
        this.beginY = begin;
        this.endY = endInclusive;
    }

    public void setImageWidth(int width) {
        this.width = width;
    }

    public void setImageHeight(int height) {
        this.height = height;
    }

    public void plot(float @NotNull [] yData, @NotNull java.awt.Color color, boolean fill) {
        int argb = color.getAlpha() << 24 | color.getRed() << 16 | color.getGreen() << 8 | color.getBlue();
        series.add(new Series(yData, argb, fill));
    }

    @NotNull
    public BufferedImage render() {
        try (Surface surface = Surface.makeRasterN32Premul(width, height)) {
            Canvas canvas = surface.getCanvas();

            float beginX = calculateBeginX();
            float endX = calculateEndX();
            float beginY = calculateBeginY();
            float endY = calculateEndY();

            float xValueRange = endX - beginX;
            float yValueRange = endY - beginY;

            float xPixelRange = width - (2 * DEFAULT_MARGIN);
            float yPixelRange = height - (2 * DEFAULT_MARGIN);

            float yPixelTop = DEFAULT_MARGIN;
            float yPixelBottom = height - DEFAULT_MARGIN;
            float xPixelStart = (float) DEFAULT_MARGIN;
            float yPixelStart = height - DEFAULT_MARGIN + (beginY * (yPixelRange / yValueRange));

            for (Series series : series) {
                int nIter = series.fill() ? 2 : 1;
                for (int iter = 0; iter < nIter; iter++) {
                    try (Path path = new Path()) {
                        float[] yData = series.data();
                        float xPos = 0, yPos;
                        for (int x = 0; x < yData.length; x++) {
                            // calculate x and y pos
                            {
                                xPos = xPixelStart + (x * (xPixelRange / xValueRange));
                                yPos = yPixelStart - (yData[x] * (yPixelRange / yValueRange));
                            }
                            if (x == 0) {
                                path.moveTo(xPos, yPos);
                            } else {
                                path.lineTo(xPos, yPos);
                            }
                        }
                        if (iter == 1 && yData.length > 0) {
                            path.lineTo(xPos, yPixelBottom);
                            path.lineTo(xPixelStart, yPixelBottom);
                            path.closePath();
                        }
                        int color = series.color();
                        try (Paint paint = new Paint()
                                .setStroke(iter == 0)
                                .setStrokeWidth(iter == 0 ? DEFAULT_STROKE_WIDTH : 0)
                                .setAntiAlias(true)) {
                            if (iter == 0) {
                                paint.setColor(color);
                            } else {
                                int colorVar1 = (color & 0x00FFFFFF) | 0x8F000000;
                                int colorVar2 = (color & 0x00FFFFFF) | 0x43000000;
                                paint.setShader(Shader.makeLinearGradient(
                                        0, yPixelTop,
                                        0, yPixelBottom,
                                        new int[]{
                                                colorVar1,
                                                colorVar2
                                        }
                                ));
                            }
                            canvas.drawPath(path, paint);
                        }
                    }
                }
            }
            renderAxis(canvas, beginX, endX, beginY, endY);

            // convert to java awt image
            {
                Image image = surface.makeImageSnapshot();
                Data data = image.encodeToData(EncodedImageFormat.PNG);
                Objects.requireNonNull(data);
                try {
                    return ImageIO.read(new ByteArrayInputStream(data.getBytes()));
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }

    public void save(@NotNull java.nio.file.Path path) {
        BufferedImage image = render();
        String extension = path.getFileName().toString().split("\\.")[1];
        try {
            ImageIO.write(image, extension, path.toFile());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Nullable
    private JFrame jFrame = null;

    public void show(boolean wait) {
        BufferedImage image = render();
        if (jFrame == null) {
            jFrame = new JFrame();
            if (name != null) {
                jFrame.setTitle(name);
            }
            jFrame.setDefaultCloseOperation(WindowConstants.HIDE_ON_CLOSE);
            jFrame.setSize(width / SCALE_FACTOR, height / SCALE_FACTOR);
            jFrame.setLocationRelativeTo(null);
        }
        jFrame.setContentPane(new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                g.drawImage(image, 0, 0, jFrame.getWidth(), jFrame.getHeight(), null);
            }
        });
        jFrame.setVisible(true);
        jFrame.repaint();
        if (wait) {
            while (jFrame.isVisible()) {
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }

    private float calculateBeginX() {
        if (beginX != null) {
            return beginX;
        }
        return 0;
    }

    private float calculateEndX() {
        if (endX != null) {
            return endX;
        }
        int maxEndX = 0;
        for (Series series : series) {
            float[] yData = series.data();
            maxEndX = Math.max(maxEndX, yData.length);
        }
        return maxEndX;
    }

    private float calculateBeginY() {
        if (beginY != null) {
            return beginY;
        }
        float minBeginY = Float.MAX_VALUE;
        for (Series series : series) {
            float[] yData = series.data();
            for (float y : yData) {
                minBeginY = Math.min(minBeginY, y);
            }
        }
        if (minBeginY == Float.MAX_VALUE) {
            return 0;
        }
        return minBeginY;
    }

    private float calculateEndY() {
        if (endY != null) {
            return endY;
        }
        float maxEndY = Float.MIN_VALUE;
        for (Series series : series) {
            float[] yData = series.data();
            for (float y : yData) {
                maxEndY = Math.max(maxEndY, y);
            }
        }
        if (maxEndY == Float.MIN_VALUE) {
            return 0;
        }
        return maxEndY;
    }

    @NotNull
    private static final Paint LINE_PAINT = new Paint();

    @NotNull
    private static final Paint DEFAULT_LABEL_PAINT = new Paint();

    @NotNull
    private static final Paint DEFAULT_NAME_TEXT_PAINT = new Paint();

    @NotNull
    private static final Typeface TYPEFACE = Objects.requireNonNull(FontMgr.getDefault().matchFamilyStyle("Roboto Mono", FontStyle.NORMAL));

    static {
        LINE_PAINT.setStrokeWidth(DEFAULT_STROKE_WIDTH);
        LINE_PAINT.setColor(0xFF000000);
        DEFAULT_LABEL_PAINT.setColor(0xFF000000);
        DEFAULT_LABEL_PAINT.setAntiAlias(true);
        DEFAULT_NAME_TEXT_PAINT.setColor(0xFF000000);
        DEFAULT_NAME_TEXT_PAINT.setAntiAlias(true);
    }

    private void renderAxis(@NotNull Canvas canvas, double beginX, double endX, double beginY, double endY) {
        // render name
        {
            String name = this.name;
            if (name != null) {
                try (Font font = new Font(TYPEFACE, DEFAULT_NAME_TEXT_FONT_SIZE)) {
                    canvas.drawString(name, width / 2f - font.measureTextWidth(name) / 2f, DEFAULT_MARGIN / 2f, font, DEFAULT_NAME_TEXT_PAINT);
                }
            }
        }

        // render X axis
        {
            canvas.drawLine(DEFAULT_MARGIN, height - DEFAULT_MARGIN, width - DEFAULT_MARGIN, height - DEFAULT_MARGIN, LINE_PAINT);
            // render label
            {
                String xLabel = this.xLabel;
                if (xLabel != null) {
                    try (Font font = new Font(TYPEFACE, DEFAULT_LABEL_FONT_SIZE)) {
                        canvas.drawString(xLabel, width / 2f - font.measureTextWidth(xLabel) / 2f, height - DEFAULT_MARGIN + font.getMetrics().getHeight(), font, DEFAULT_LABEL_PAINT);
                    }
                }
            }
        }
        // render Y axis
        {
            canvas.drawLine(DEFAULT_MARGIN, DEFAULT_MARGIN, DEFAULT_MARGIN, height - DEFAULT_MARGIN, LINE_PAINT);
            // render label
            {
                String yLabel = this.yLabel;
                if (yLabel != null) {
                    try (Font font = new Font(TYPEFACE, DEFAULT_LABEL_FONT_SIZE)) {
                        float yLabelWidth = font.measureTextWidth(yLabel);
                        canvas.save();
                        canvas.translate(DEFAULT_MARGIN - font.getMetrics().getHeight() / 2f, height / 2f);
                        canvas.rotate(-90);
                        canvas.drawString(yLabel, -yLabelWidth / 2f, 0, font, DEFAULT_LABEL_PAINT);
                        canvas.restore();
                    }
                }
            }
        }
    }
}
