package me.mikex86.scicore.graphviz;

import me.mikex86.scicore.op.Graph;
import me.mikex86.scicore.op.IGraph;
import org.jetbrains.annotations.NotNull;

import java.util.*;

public class GraphRenderPlan {

    @NotNull
    private final RowStack rowStack;

    public GraphRenderPlan(@NotNull RowStack rowStack) {
        this.rowStack = rowStack;
    }

    public int getNumRows() {
        return rowStack.getNumRows();
    }

    public int getMaxNumNodesPerRow() {
        return rowStack.getMaxNumNodes();
    }

    @NotNull
    public List<Row> getRows() {
        return rowStack.getRows();
    }

    @NotNull
    public static GraphRenderPlan makeRenderPlan(@NotNull IGraph graph) {
        RowStack rowStack = new RowStack();
        int rowIndex = 0;

        IGraph.IGraphNode currentNode = graph.getOutputNode();

        // add current node
        {
            Column column = new Column(List.of(currentNode));
            rowStack.getRow(rowIndex).add(column);
        }

        makeRenderPlan(rowStack, currentNode, rowIndex + 1);
        return new GraphRenderPlan(rowStack);
    }

    private static void makeRenderPlan(@NotNull RowStack rowStack, @NotNull IGraph.IGraphNode currentNode, int rowIndex) {
        if (currentNode instanceof Graph.OperationGraphNode operationNode) {
            makeOperationNodeRenderPlan(rowStack, operationNode, rowIndex);
        }
    }

    private static void makeOperationNodeRenderPlan(@NotNull RowStack rowStack, @NotNull Graph.OperationGraphNode currentNode, int rowIndex) {
        // add inputs
        {
            List<IGraph.IGraphNode> inputs = currentNode.getInputs();

            // Cluster the input nodes for the current node as a column in the current row
            {
                Column column = new Column(Collections.unmodifiableList(inputs));
                rowStack.getRow(rowIndex).add(column);
            }

            // Recursive render planning for inputs. The inputs nodes themselves are already in the plan,
            // now we just need recursively add the inputs of the inputs in the next row.
            // Note this recursion is in a loop. The recursion will back off to rowIndex + 1
            // potentially multiple times if inputs.size() > 0. This ensures that multiple columns are placed in the
            // same row, such that their row is equal to the node's graph depth in the graph.
            for (IGraph.IGraphNode input : inputs) {
                makeRenderPlan(rowStack, input, rowIndex + 1);
            }
        }
    }

    public static class RowStack {

        @NotNull
        private final List<Row> rows = new LinkedList<>();

        @NotNull
        public Row getRow(int rowIndex) {
            if (rowIndex == rows.size()) {
                Row row = new Row();
                rows.add(row);
                return row;
            } else if (rowIndex < rows.size()) {
                return rows.get(rowIndex);
            } else {
                throw new IllegalArgumentException("Row index accessed row stack in a non-contiguous manner");
            }
        }

        public int getNumRows() {
            return rows.size();
        }

        public int getMaxNumNodes() {
            int maxNumNodes = 0;

            for (Row row : rows) {
                maxNumNodes = Math.max(maxNumNodes, row.getMaxNumNodes());
            }

            return maxNumNodes;
        }

        @NotNull
        public List<Row> getRows() {
            return rows;
        }
    }

    /**
     * Represents a row of the render plan. It contains 'columns', a list of nodes that are rendered close to each other.
     * Columns are separated by space in the final graph render.
     */
    public record Row(@NotNull List<Column> columns) {

        public Row() {
            this(new LinkedList<>());
        }

        public void add(@NotNull Column column) {
            columns.add(column);
        }

        public int getMaxNumNodes() {
            int maxNumNodes = 0;

            for (Column column : columns) {
                maxNumNodes = Math.max(maxNumNodes, column.getNumNodes());
            }

            return maxNumNodes;
        }
    }

    /**
     * Represents a column of nodes that are rendered close to each other.
     *
     * @param nodes the nodes of the graph
     */
    public record Column(@NotNull List<IGraph.IGraphNode> nodes) {

        public int getNumNodes() {
            return nodes.size();
        }

    }

}
