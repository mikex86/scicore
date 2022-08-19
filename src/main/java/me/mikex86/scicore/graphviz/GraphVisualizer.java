package me.mikex86.scicore.graphviz;

import me.mikex86.scicore.op.Graph;
import me.mikex86.scicore.op.IGraph;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.skija.*;

import java.util.LinkedHashMap;

public class GraphVisualizer {


    private static final int GRAPH_SCALE = 2;

    private static final int TENSOR_NODE_HEAD_WIDTH = 400 * GRAPH_SCALE;

    private static final int TENSOR_NODE_HEAD_HEIGHT = 400 * GRAPH_SCALE;

    private static final int OPERATION_NODE_RADIUS = 250 * GRAPH_SCALE;

    private static final int NODE_INTERCONNECT_SPACE_HEIGHT = 140 * GRAPH_SCALE;

    private static final int WIDTH_PER_NODE = TENSOR_NODE_HEAD_WIDTH;

    private static final int HEIGHT_PER_ROW = TENSOR_NODE_HEAD_HEIGHT + NODE_INTERCONNECT_SPACE_HEIGHT;

    private static final int COLUMN_BACKGROUND_BORDER_RADIUS = 15 * GRAPH_SCALE;

    private static final int NODE_BACKGROUND_BORDER_RADIUS = 12 * GRAPH_SCALE;

    private static final int COLUMN_PADDING = 10 * GRAPH_SCALE;

    private static final int NODE_PADDING = 10 * GRAPH_SCALE;

    private static final int NODE_HEADING_FONT_SIZE = 18 * GRAPH_SCALE;

    private static final int NODE_MAIN_FONT_SIZE = 13 * GRAPH_SCALE;

    private static final int NODE_HEADING_TEXT_PADDING = 10 * GRAPH_SCALE;

    private static final int NODE_MAIN_TEXT_PADDING = 10 * GRAPH_SCALE;

    private static final int NODE_MAIN_TEXT_BACKGROUND_BORDER_RADIUS = 8 * GRAPH_SCALE;

    @NotNull
    private static final Typeface TYPEFACE = FontMgr.getDefault().matchFamilyStyle("Roboto", FontStyle.NORMAL);

    @NotNull
    private static final Paint COLUM_BACKGROUND_COLOR = new Paint();

    @NotNull
    private static final Paint TENSOR_NODE_COLOR = new Paint();

    @NotNull
    private static final Paint TENSOR_NODE_COLOR_STROKE = new Paint();

    @NotNull
    private static final Paint TENSOR_NODE_BACKGROUND_COLOR = new Paint();

    @NotNull
    private static final Paint OPERATION_NODE_COLOR = new Paint();

    @NotNull
    private static final Paint HEADING_TEXT_COLOR = new Paint();

    @NotNull
    private static final Paint LINE_PAINT = new Paint();

    private static final int NODE_BORDER_WIDTH = 4 * GRAPH_SCALE;

    static {
        COLUM_BACKGROUND_COLOR.setColor(0xFFafafaf);
        TENSOR_NODE_COLOR.setColor(0xff0984e3);
        TENSOR_NODE_COLOR_STROKE.setColor(0xff0984e3);
        TENSOR_NODE_COLOR_STROKE.setStrokeWidth(NODE_BORDER_WIDTH);
        TENSOR_NODE_COLOR_STROKE.setMode(PaintMode.STROKE);
        OPERATION_NODE_COLOR.setColor(0xffd63031);
        HEADING_TEXT_COLOR.setColor(0xFFFFFFFF);
        TENSOR_NODE_BACKGROUND_COLOR.setColor(0xFF2c3e50);
        LINE_PAINT.setColor(0xFF000000);
        LINE_PAINT.setMode(PaintMode.STROKE);
        LINE_PAINT.setStrokeWidth(2 * GRAPH_SCALE);
    }

    @NotNull
    public static Image visualizeGraph(@NotNull IGraph graph) {
        GraphRenderPlan renderPlan = GraphRenderPlan.makeRenderPlan(graph);

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

            return surface.makeImageSnapshot();
        }
    }

    private static void renderInterconnect(@NotNull GraphRenderPlan.Row row, @NotNull GraphRenderPlan.Row nextRow, int rowIndex, @NotNull Canvas canvas) {
        int y = rowIndex * HEIGHT_PER_ROW;
        int nextY = (rowIndex - 1) * HEIGHT_PER_ROW;
        for (GraphRenderPlan.Column column : row.columns()) {
            int nodeIndex = 0;
            for (IGraph.IGraphNode node : column.nodes()) {
                int x = nodeIndex * WIDTH_PER_NODE;
                if (node instanceof Graph.OperationGraphNode operationGraphNode) {
                    for (IGraph.IGraphNode input : operationGraphNode.getInputs()) {
                        int nextX = getXPosition(input, nextRow);
                        try (Path path = new Path()) {
                            path.moveTo(x + WIDTH_PER_NODE / 2f, y + HEIGHT_PER_ROW / 2f);
//                            path.lineTo(nextX + WIDTH_PER_NODE / 2f, nextY + HEIGHT_PER_ROW / 2f);
                            path.cubicTo(
                                    x + WIDTH_PER_NODE / 2f, y + HEIGHT_PER_ROW / 2f,
                                    nextX + WIDTH_PER_NODE / 2f, y,
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

    private static int getXPosition(@NotNull IGraph.IGraphNode input, @NotNull GraphRenderPlan.Row nextRow) {
        int x = 0;
        for (GraphRenderPlan.Column column : nextRow.columns()) {
            for (IGraph.IGraphNode node : column.nodes()) {
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
            for (IGraph.IGraphNode node : column.nodes()) {
                x = renderNode(node, x, y, canvas);
            }
        }
        return x;
    }

    private static int renderNode(@NotNull IGraph.IGraphNode node, int x, int y, @NotNull Canvas canvas) {
        // render node
        if (node instanceof Graph.TensorDeclarationGraphNode tensorNode) {
            return renderTensorNode(tensorNode, x, y, canvas);
        } else if (node instanceof Graph.OperationGraphNode) {
            return renderOperationNode((Graph.OperationGraphNode) node, x, y, canvas);
        } else {
            throw new IllegalArgumentException("Unknown node type: " + node.getClass().getName());
        }
    }

    private static int renderTensorNode(@NotNull Graph.TensorDeclarationGraphNode node, int x, int y, Canvas canvas) {
        return renderNode(node.getName() + " (Tensor)", new LinkedHashMap<>(), x, y, canvas);
    }

    private static int renderNode(@NotNull String title, @NotNull LinkedHashMap<String, String> attributes, int x, int y, Canvas canvas) {
        try (Font headingFont = new Font(TYPEFACE, NODE_HEADING_FONT_SIZE)) {

            // render background
            {
                // render background
                RRect rect = RRect.makeLTRB(
                        x + COLUMN_PADDING + NODE_PADDING + NODE_BORDER_WIDTH / 2f,
                        y + COLUMN_PADDING + NODE_PADDING + NODE_BORDER_WIDTH / 2f,
                        x + TENSOR_NODE_HEAD_WIDTH - NODE_PADDING - COLUMN_PADDING - NODE_BORDER_WIDTH / 2f,
                        y + TENSOR_NODE_HEAD_HEIGHT - NODE_PADDING - COLUMN_PADDING - NODE_BORDER_WIDTH / 2f,
                        NODE_BACKGROUND_BORDER_RADIUS
                );
                canvas.drawRRect(rect, TENSOR_NODE_BACKGROUND_COLOR);

                // render top bar
                rect = RRect.makeLTRB(
                        x + COLUMN_PADDING + NODE_PADDING,
                        y + COLUMN_PADDING + NODE_PADDING,
                        x + TENSOR_NODE_HEAD_WIDTH - NODE_PADDING - COLUMN_PADDING,
                        y + COLUMN_PADDING + NODE_PADDING + NODE_HEADING_TEXT_PADDING + (headingFont.getMetrics().getBottom() - headingFont.getMetrics().getTop()) + NODE_HEADING_TEXT_PADDING,
                        NODE_BACKGROUND_BORDER_RADIUS, NODE_BACKGROUND_BORDER_RADIUS, 0, 0
                );
                canvas.drawRRect(rect, TENSOR_NODE_COLOR);

                // render border
                rect = RRect.makeLTRB(
                        x + COLUMN_PADDING + NODE_PADDING + NODE_BORDER_WIDTH / 2f,
                        y + COLUMN_PADDING + NODE_PADDING + NODE_BORDER_WIDTH / 2f,
                        x + TENSOR_NODE_HEAD_WIDTH - NODE_PADDING - COLUMN_PADDING - NODE_BORDER_WIDTH / 2f,
                        y + TENSOR_NODE_HEAD_HEIGHT - NODE_PADDING - COLUMN_PADDING - NODE_BORDER_WIDTH / 2f,
                        NODE_BACKGROUND_BORDER_RADIUS - 10f
                );
                canvas.drawRRect(rect, TENSOR_NODE_COLOR_STROKE);
            }

            // render heading
            canvas.drawString(
                    title,
                    x + COLUMN_PADDING + NODE_PADDING + NODE_HEADING_TEXT_PADDING,
                    y + COLUMN_PADDING + NODE_PADDING + NODE_HEADING_TEXT_PADDING - headingFont.getMetrics().getTop(),
                    headingFont,
                    HEADING_TEXT_COLOR
            );
        }
        return x + WIDTH_PER_NODE;
    }

    public static int renderOperationNode(@NotNull Graph.OperationGraphNode node, int x, int y, Canvas canvas) {
        // render background
        canvas.drawCircle(
                x + WIDTH_PER_NODE / 2f,
                y + TENSOR_NODE_HEAD_WIDTH / 2f,
                OPERATION_NODE_RADIUS / 2f - COLUMN_PADDING - NODE_PADDING,
                OPERATION_NODE_COLOR
        );

        // render text
        try (Font font = new Font(TYPEFACE, NODE_HEADING_FONT_SIZE)) {
            canvas.drawString(
                    node.getName(),
                    x + WIDTH_PER_NODE / 2f - font.measureTextWidth(node.getName()) / 2f,
                    y + TENSOR_NODE_HEAD_WIDTH / 2f + font.getMetrics().getBottom(),
                    font,
                    HEADING_TEXT_COLOR
            );
        }
        return x + WIDTH_PER_NODE;
    }

}
