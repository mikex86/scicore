package me.mikex86.matplotlib.graphviz;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.skija.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.Objects;

public class GraphVisualizer {

    private static final int GRAPH_SCALE = 2;

    private static final int DATA_NODE_HEAD_WIDTH = 200 * GRAPH_SCALE;

    private static final int DATA_NODE_HEAD_HEIGHT = 145 * GRAPH_SCALE;

    private static final int INTERCONNECT_NODE_RADIUS = 150 * GRAPH_SCALE;

    private static final int NODE_INTERCONNECT_SPACE_HEIGHT = 80 * GRAPH_SCALE;

    private static final int WIDTH_PER_NODE = DATA_NODE_HEAD_WIDTH;

    private static final int HEIGHT_PER_ROW = DATA_NODE_HEAD_HEIGHT + NODE_INTERCONNECT_SPACE_HEIGHT;

    private static final int COLUMN_BACKGROUND_BORDER_RADIUS = 15 * GRAPH_SCALE;

    private static final int NODE_BACKGROUND_BORDER_RADIUS = 12 * GRAPH_SCALE;

    private static final int COLUMN_PADDING = 10 * GRAPH_SCALE;

    private static final int NODE_PADDING = 10 * GRAPH_SCALE;

    private static final int NODE_HEADING_FONT_SIZE = 18 * GRAPH_SCALE;

    private static final int NODE_ATTRIBUTE_FONT_SIZE = 14 * GRAPH_SCALE;

    private static final int NODE_HEADING_TEXT_PADDING = 10 * GRAPH_SCALE;

    private static final int NODE_ATTRIBUTE_TEXT_PADDING = 10 * GRAPH_SCALE;

    private static final int NODE_MAIN_TEXT_BACKGROUND_BORDER_RADIUS = 8 * GRAPH_SCALE;

    @NotNull
    private static final Typeface HEADING_TYPEFACE = Objects.requireNonNullElse(FontMgr.getDefault().matchFamilyStyle("Roboto", FontStyle.NORMAL), Typeface.makeDefault());

    @NotNull
    private static final Typeface ATTRIBUTE_TYPEFACE = Objects.requireNonNullElse(FontMgr.getDefault().matchFamilyStyle("Roboto Mono", FontStyle.NORMAL), Typeface.makeDefault());

    @NotNull
    private static final Paint COLUM_BACKGROUND_COLOR = new Paint();

    @NotNull
    private static final Paint DATA_NODE_COLOR = new Paint();

    @NotNull
    private static final Paint DATA_NODE_COLOR_STROKE = new Paint();

    @NotNull
    private static final Paint DATA_NODE_BACKGROUND_COLOR = new Paint();

    @NotNull
    private static final Paint OPERATION_NODE_COLOR = new Paint();

    @NotNull
    private static final Paint HEADING_TEXT_COLOR = new Paint();

    @NotNull
    private static final Paint ATTRIBUTE_TEXT_COLOR = new Paint();

    @NotNull
    private static final Paint LINE_PAINT = new Paint();

    private static final int NODE_BORDER_WIDTH = 4 * GRAPH_SCALE;

    static {
        COLUM_BACKGROUND_COLOR.setColor(0xFFafafaf);
        DATA_NODE_COLOR.setColor(0xff0984e3);
        DATA_NODE_COLOR_STROKE.setColor(0xff0984e3);
        DATA_NODE_COLOR_STROKE.setStrokeWidth(NODE_BORDER_WIDTH);
        DATA_NODE_COLOR_STROKE.setMode(PaintMode.STROKE);
        OPERATION_NODE_COLOR.setColor(0xffd63031);
        HEADING_TEXT_COLOR.setColor(0xFFFFFFFF);
        ATTRIBUTE_TEXT_COLOR.setColor(0xFFFBFBFB);
        DATA_NODE_BACKGROUND_COLOR.setColor(0xFF2c3e50);
        LINE_PAINT.setColor(0xFF000000);
        LINE_PAINT.setMode(PaintMode.STROKE);
        LINE_PAINT.setStrokeWidth(2 * GRAPH_SCALE);
    }

    @NotNull
    public static BufferedImage visualizeGraph(@NotNull GraphRenderPlan renderPlan) {

        try (Surface surface = Surface.makeRasterN32Premul(renderPlan.getMaxNumNodesPerRow() * WIDTH_PER_NODE, renderPlan.getNumRows() * HEIGHT_PER_ROW)) {
            Canvas canvas = surface.getCanvas();

            // render rows (reversed because output node is row 0, but should be rendered last)
            {
                int nRows = renderPlan.getNumRows();

                // render interconnect lines
                for (int i = 0; i < nRows; i++) {
                    GraphRenderPlan.Row row = renderPlan.getRows().get(nRows - i - 1);

                    // if has next row
                    if (nRows - i < nRows) {
                        GraphRenderPlan.Row nextRow = renderPlan.getRows().get(nRows - i);
                        renderInterconnect(row, nextRow, i, canvas);
                    }
                }

                // render main nodes
                for (int i = 0; i < nRows; i++) {
                    GraphRenderPlan.Row row = renderPlan.getRows().get(nRows - i - 1);
                    renderRow(row, i, canvas);
                }
            }

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

    public static void saveGraph(@NotNull GraphRenderPlan renderPlan, @NotNull String filename) {
        BufferedImage image = visualizeGraph(renderPlan);
        try {
            ImageIO.write(image, "png", new File(filename));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static void renderInterconnect(@NotNull GraphRenderPlan.Row row, @NotNull GraphRenderPlan.Row nextRow, int rowIndex, @NotNull Canvas canvas) {
        int y = rowIndex * HEIGHT_PER_ROW;
        int nextY = (rowIndex - 1) * HEIGHT_PER_ROW;
        for (GraphRenderPlan.Column column : row.columns()) {
            int nodeIndex = 0;
            for (GraphRenderPlan.IGraphNode node : column.nodes()) {
                int x = nodeIndex * WIDTH_PER_NODE;
                if (node instanceof GraphRenderPlan.IGraphNode.Interconnect interconnect) {
                    for (GraphRenderPlan.IGraphNode input : interconnect.incomingNodes()) {
                        int nextX = getXPosition(input, nextRow);
                        try (Path path = new Path()) {
                            path.moveTo(x + WIDTH_PER_NODE / 2f, y + HEIGHT_PER_ROW / 2f);
//                            path.lineTo(nextX + WIDTH_PER_NODE / 2f, nextY + HEIGHT_PER_ROW / 2f);
                            path.cubicTo(
                                    x + WIDTH_PER_NODE / 2f, y + HEIGHT_PER_ROW / 2f,
                                    nextX + WIDTH_PER_NODE / 2f, y - HEIGHT_PER_ROW / 8f,
                                    nextX + WIDTH_PER_NODE / 2f, nextY + HEIGHT_PER_ROW / 2f
                            );
                            canvas.drawPath(path, LINE_PAINT);
                        }
                    }
                }
                nodeIndex++;
            }
        }
    }

    private static int getXPosition(@NotNull GraphRenderPlan.IGraphNode input, @NotNull GraphRenderPlan.Row nextRow) {
        int x = 0;
        for (GraphRenderPlan.Column column : nextRow.columns()) {
            for (GraphRenderPlan.IGraphNode node : column.nodes()) {
                if (node == input) {
                    return x;
                }
                x += WIDTH_PER_NODE;
            }
        }
        return x;
    }

    private static void renderRow(@NotNull GraphRenderPlan.Row row, int rowIndex, @NotNull Canvas canvas) {
        int x = 0;
        for (GraphRenderPlan.Column column : row.columns()) {
            x = renderColumn(column, x, rowIndex * HEIGHT_PER_ROW, canvas);
        }
    }

    private static int renderColumn(@NotNull GraphRenderPlan.Column column, int x, int y, @NotNull Canvas canvas) {
        int columWidth = column.getNumNodes() * WIDTH_PER_NODE;

        // render column background
        {
//            RRect rect = RRect.makeLTRB(
//                    x + COLUMN_PADDING,
//                    y + COLUMN_PADDING,
//                    x + columWidth - COLUMN_PADDING,
//                    y + TENSOR_NODE_HEAD_HEIGHT - COLUMN_PADDING,
//                    COLUMN_BACKGROUND_BORDER_RADIUS
//            );
//            canvas.drawRRect(rect, COLUM_BACKGROUND_COLOR);
        }

        // render nodes
        {
            for (GraphRenderPlan.IGraphNode node : column.nodes()) {
                x = renderNode(node, x, y, canvas);
            }
        }
        return x;
    }

    private static int renderNode(@NotNull GraphRenderPlan.IGraphNode node, int x, int y, @NotNull Canvas canvas) {
        // render node
        if (node instanceof GraphRenderPlan.IGraphNode.DataNode dataNode) {
            return renderDataNode(dataNode, x, y, canvas);
        } else if (node instanceof GraphRenderPlan.IGraphNode.Interconnect interconnect) {
            return renderInterconnect(interconnect, x, y, canvas);
        } else {
            throw new IllegalArgumentException("Unknown node type: " + node.getClass().getName());
        }
    }

    private static int renderDataNode(@NotNull GraphRenderPlan.IGraphNode.DataNode node, int x, int y, Canvas canvas) {
        return renderNode(node.name(), node.attributes(), x, y, canvas);
    }

    private static int renderNode(@NotNull String title, @NotNull Map<String, String> attributes, int x, int y, Canvas canvas) {
        try (Font headingFont = new Font(HEADING_TYPEFACE, NODE_HEADING_FONT_SIZE)) {

            // render background
            {
                // render background
                RRect rect = RRect.makeLTRB(
                        x + COLUMN_PADDING + NODE_PADDING + NODE_BORDER_WIDTH / 2f,
                        y + COLUMN_PADDING + NODE_PADDING + NODE_BORDER_WIDTH / 2f,
                        x + DATA_NODE_HEAD_WIDTH - NODE_PADDING - COLUMN_PADDING - NODE_BORDER_WIDTH / 2f,
                        y + DATA_NODE_HEAD_HEIGHT - NODE_PADDING - COLUMN_PADDING - NODE_BORDER_WIDTH / 2f,
                        NODE_BACKGROUND_BORDER_RADIUS
                );
                canvas.drawRRect(rect, DATA_NODE_BACKGROUND_COLOR);

                // render top bar
                rect = RRect.makeLTRB(
                        x + COLUMN_PADDING + NODE_PADDING,
                        y + COLUMN_PADDING + NODE_PADDING,
                        x + DATA_NODE_HEAD_WIDTH - NODE_PADDING - COLUMN_PADDING,
                        y + COLUMN_PADDING + NODE_PADDING + NODE_HEADING_TEXT_PADDING + (headingFont.getMetrics().getBottom() - headingFont.getMetrics().getTop()) + NODE_HEADING_TEXT_PADDING,
                        NODE_BACKGROUND_BORDER_RADIUS, NODE_BACKGROUND_BORDER_RADIUS, 0, 0
                );
                canvas.drawRRect(rect, DATA_NODE_COLOR);

                // render border
                rect = RRect.makeLTRB(
                        x + COLUMN_PADDING + NODE_PADDING + NODE_BORDER_WIDTH / 2f,
                        y + COLUMN_PADDING + NODE_PADDING + NODE_BORDER_WIDTH / 2f,
                        x + DATA_NODE_HEAD_WIDTH - NODE_PADDING - COLUMN_PADDING - NODE_BORDER_WIDTH / 2f,
                        y + DATA_NODE_HEAD_HEIGHT - NODE_PADDING - COLUMN_PADDING - NODE_BORDER_WIDTH / 2f,
                        NODE_BACKGROUND_BORDER_RADIUS - 10f
                );
                canvas.drawRRect(rect, DATA_NODE_COLOR_STROKE);
            }

            // render heading
            canvas.drawString(
                    title,
                    x + COLUMN_PADDING + NODE_PADDING + NODE_HEADING_TEXT_PADDING,
                    y + COLUMN_PADDING + NODE_PADDING + NODE_HEADING_TEXT_PADDING - headingFont.getMetrics().getTop(),
                    headingFont,
                    HEADING_TEXT_COLOR
            );

            try (Font attributeFont = new Font(ATTRIBUTE_TYPEFACE, NODE_ATTRIBUTE_FONT_SIZE)) {
                // render attributes
                float yOffset = y + COLUMN_PADDING + NODE_PADDING + NODE_HEADING_TEXT_PADDING + (headingFont.getMetrics().getBottom() - headingFont.getMetrics().getTop()) + NODE_HEADING_TEXT_PADDING;
                for (Map.Entry<String, String> attribute : attributes.entrySet()) {
                    canvas.drawString(
                            attribute.getKey() + ": " + attribute.getValue(),
                            x + COLUMN_PADDING + NODE_PADDING + NODE_ATTRIBUTE_TEXT_PADDING,
                            yOffset - attributeFont.getMetrics().getTop(),
                            attributeFont,
                            ATTRIBUTE_TEXT_COLOR
                    );
                    yOffset += attributeFont.getMetrics().getHeight();
                }
            }
        }
        return x + WIDTH_PER_NODE;
    }

    public static int renderInterconnect(@NotNull GraphRenderPlan.IGraphNode.Interconnect node, int x, int y, Canvas canvas) {
        // render background
        canvas.drawCircle(
                x + WIDTH_PER_NODE / 2f,
                y + DATA_NODE_HEAD_WIDTH / 2f,
                INTERCONNECT_NODE_RADIUS / 2f - COLUMN_PADDING - NODE_PADDING,
                OPERATION_NODE_COLOR
        );

        // render text
        try (Font font = new Font(HEADING_TYPEFACE, NODE_HEADING_FONT_SIZE)) {
            canvas.drawString(
                    node.name(),
                    x + WIDTH_PER_NODE / 2f - font.measureTextWidth(node.name()) / 2f,
                    y + DATA_NODE_HEAD_WIDTH / 2f + font.getMetrics().getBottom(),
                    font,
                    HEADING_TEXT_COLOR
            );
        }
        return x + WIDTH_PER_NODE;
    }

}
