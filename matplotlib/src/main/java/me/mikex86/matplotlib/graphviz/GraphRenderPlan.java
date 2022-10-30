package me.mikex86.matplotlib.graphviz;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

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

            // compare via identity

            @Override
            public boolean equals(Object obj) {
                return obj == this;
            }

            @Override
            public int hashCode() {
                return System.identityHashCode(this);
            }
        }

        record DataNode(@NotNull String name, @NotNull Map<String, String> attributes) implements IGraphNode {

            // compare via identity

            @Override
            public boolean equals(Object obj) {
                return obj == this;
            }

            @Override
            public int hashCode() {
                return System.identityHashCode(this);
            }
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
                maxNumNodes = Math.max(maxNumNodes, row.getNumNodes());
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
    public static class Row {

        @NotNull
        private final List<Column> columns;

        public Row(@NotNull List<Column> columns) {
            this.columns = columns;
        }

        public Row() {
            this(new LinkedList<>());
        }

        public void add(@NotNull Column column) {
            columns.add(column);
        }

        @NotNull
        public List<Column> columns() {
            return columns;
        }

        public int getNumNodes() {
            int numNodes = 0;

            for (Column column : columns) {
                numNodes += column.getNumNodes();
            }

            return numNodes;
        }

        /**
         * Removes the node from the column in this row that contains it.
         *
         * @param graphNode The node to remove.
         * @return the column instance that contained the node, or null if the node was not found.
         */
        @Nullable
        public Column remove(@NotNull IGraphNode graphNode) {
            for (Column column : columns) {
                if (column.removeNode(graphNode)) {
                    if (column.nodes().isEmpty()) {
                        columns.remove(column);
                    }
                    return column;
                }
            }
            return null;
        }

        @NotNull
        private final Map<Column, Column> columnReplacementMap = new HashMap<>();

        /**
         * Adds the specified graph node to a column in this row.
         * The exact column that the node is added to is determined by the identity of the specified column.
         * Calling this method with the same column instance multiple times will add the node to the same column.
         * The column instance itself is not modified by this method.
         *
         * @param graphNode the node to add.
         * @param column    the column whose identity determines the column that the node is added to.
         */
        public void addNodeFromColumn(@NotNull IGraphNode graphNode, @NotNull Column column) {
            Column replacementColumn = columnReplacementMap.get(column);
            if (replacementColumn == null) {
                replacementColumn = new Column(new ArrayList<>());
                columnReplacementMap.put(column, replacementColumn);
                columns.add(replacementColumn);
            }
            replacementColumn.addNode(graphNode);
        }

        @Override
        public String toString() {
            return "Row{" +
                   "columns=" + columns +
                   '}';
        }
    }

    /**
     * Represents a column of nodes that are rendered close to each other.
     */
    public static final class Column {

        private final @NotNull List<IGraphNode> nodes;

        /**
         * @param nodes the nodes of the graph
         */
        public Column(@NotNull List<IGraphNode> nodes) {
            this.nodes = new ArrayList<>(nodes);
        }

        public int getNumNodes() {
            return nodes.size();
        }

        public @NotNull List<IGraphNode> nodes() {
            return nodes;
        }

        public void addNode(@NotNull IGraphNode graphNode) {
            nodes.add(graphNode);
        }

        public boolean removeNode(@NotNull IGraphNode node) {
            return nodes.remove(node);
        }

        @Override
        public String toString() {
            return "Column{" +
                   "nodes=" + nodes +
                   '}';
        }
    }

}
