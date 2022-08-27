package me.mikex86.matplotlib.graphviz;

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

    public interface IGraphNode {

        record Interconnect(@NotNull String name, @NotNull List<IGraphNode> incomingNodes) implements IGraphNode {
        }

        record DataNode(@NotNull String name, @NotNull Map<String, String> attributes) implements IGraphNode {
        }

        @NotNull String name();

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
    public record Column(@NotNull List<IGraphNode> nodes) {

        public int getNumNodes() {
            return nodes.size();
        }

    }

}
