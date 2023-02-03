package me.mikex86.scicore.graphviz;

import me.mikex86.matplotlib.graphviz.GraphRenderPlan;
import me.mikex86.matplotlib.graphviz.GraphVisualizer;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.awt.image.BufferedImage;
import java.util.*;
import java.util.List;

public class DAGGraphRenderPlanFactory {

    @NotNull
    public static GraphRenderPlan makeRenderPlan(@NotNull IGraph graph) {
        GraphRenderPlan.RowStack rowStack = new GraphRenderPlan.RowStack();

        IGraph.IGraphNode rootNode = graph.getOutputNode();

        // reverse topology
        Queue<IGraph.IGraphNode> toVisit = new LinkedList<>();
        Deque<Graph.OperationGraphNode> reverseTopology = new LinkedList<>();

        toVisit.add(rootNode);

        while (!toVisit.isEmpty()) {
            IGraph.IGraphNode node = toVisit.remove();
            if (node instanceof Graph.OperationGraphNode operationGraphNode) {
                toVisit.addAll(operationGraphNode.getInputs());

                reverseTopology.remove(operationGraphNode); // remove previous occurrences
                reverseTopology.addFirst(operationGraphNode);
            }

        }

        Map<IGraph.IGraphNode, GraphRenderPlan.IGraphNode> graphNodes = new IdentityHashMap<>();
        Map<GraphRenderPlan.IGraphNode, Integer> nodeToRowIndex = new IdentityHashMap<>();

        // stores in which row index the respective node is first used (for interconnects)
        Map<GraphRenderPlan.IGraphNode, Integer> nodeToFirstUsageRowIndex = new IdentityHashMap<>();

        // create graph nodes
        for (Graph.OperationGraphNode node : reverseTopology) {
            List<IGraph.IGraphNode> inputs = node.getInputs();
            List<GraphRenderPlan.IGraphNode> graphInputs = new ArrayList<>(inputs.size());
            List<GraphRenderPlan.IGraphNode> newGraphNodes = new ArrayList<>(inputs.size());
            int highestRowWhereInputUsed = 0;
            for (IGraph.IGraphNode input : inputs) {
                GraphRenderPlan.IGraphNode existingNode = graphNodes.get(input);
                if (existingNode == null) {
                    if (input instanceof Graph.TensorDeclarationGraphNode tensorDeclarationGraphNode) {
                        GraphRenderPlan.IGraphNode.DataNode dataNode = toGraphNode(tensorDeclarationGraphNode, graph);
                        graphNodes.put(input, dataNode);
                        graphInputs.add(dataNode);
                        nodeToRowIndex.put(dataNode, 0);
                        newGraphNodes.add(dataNode);
                    } else if (input instanceof Graph.OperationGraphNode) {
                        throw new IllegalStateException("Input node to operation is itself an operation, but was not found in the list of already created render graph nodes." +
                                                        "The current operation depends on said operation, " +
                                                        "thus should have been evaluated first and thus higher up in the graph, thus already in this list.");
                    }
                } else {
                    highestRowWhereInputUsed = Math.max(highestRowWhereInputUsed, nodeToRowIndex.get(existingNode));
                    graphInputs.add(existingNode);
                }
            }

            int currentOperationRowIndex = highestRowWhereInputUsed + 1;

            // populate nodeToFirstUsageRowIndex
            for (GraphRenderPlan.IGraphNode graphNode : graphInputs) {
                if (nodeToFirstUsageRowIndex.containsKey(graphNode)) {
                    continue;
                }
                nodeToFirstUsageRowIndex.put(graphNode, currentOperationRowIndex);
            }

//            // move inputs one above their first usage row
//            for (GraphRenderPlan.IGraphNode graphNode : graphInputs) {
//                int firstUsageRowIndex = nodeToFirstUsageRowIndex.get(graphNode);
//                int couldBeRowIndex = firstUsageRowIndex - 1;
//                int currentRowIndexOfNode = nodeToRowIndex.get(graphNode);
//                if (couldBeRowIndex > currentRowIndexOfNode) {
//                    nodeToRowIndex.put(graphNode, couldBeRowIndex);
//                    GraphRenderPlan.Column columnThatContainedTheRow = rowStack.getRow(currentRowIndexOfNode).remove(graphNode); // remove from old row
//                    if (columnThatContainedTheRow == null) {
//                        continue; // no need to move, this node is yet to be added
//                    }
//                    rowStack.getRow(couldBeRowIndex).addNodeFromColumn(graphNode, columnThatContainedTheRow); // add to new row
//                }
//            }

            // create render graph nodes for the inputs
            if (!newGraphNodes.isEmpty()) {
                GraphRenderPlan.Column column = new GraphRenderPlan.Column(newGraphNodes);
                GraphRenderPlan.Row row = rowStack.getRow(highestRowWhereInputUsed);
                row.add(column);
            }

            // create render graph node for the operation
            {
                GraphRenderPlan.IGraphNode.Interconnect interconnect = new GraphRenderPlan.IGraphNode.Interconnect(node.getName(), graphInputs);
                graphNodes.put(node, interconnect);
                nodeToRowIndex.put(interconnect, currentOperationRowIndex);
                rowStack.getRow(currentOperationRowIndex).add(new GraphRenderPlan.Column(List.of(interconnect)));
            }
        }

        return new GraphRenderPlan(rowStack);
    }

    private static GraphRenderPlan.IGraphNode.@NotNull DataNode toGraphNode(@NotNull Graph.TensorDeclarationGraphNode tensorNode, @NotNull IGraph graph) {
        ITensor value = tensorNode.getValue();
        Map<String, String> attributes = new LinkedHashMap<>();
        attributes.put("dataType", value.getDataType().toString());
        attributes.put("shape", ShapeUtils.toString(value.getShape()));
        attributes.put("isScalar", Boolean.toString(value.isScalar()));
        attributes.put("backend", value.getSciCoreBackend().getBackendType().name());
        //graph.getGradient(value).ifPresent(gradient -> attributes.put("gradient", gradient.toString()));
        if (value.isScalar()) {
            attributes.put("value", value.element(Object.class).toString());
        }
        return new GraphRenderPlan.IGraphNode.DataNode(tensorNode.getName() + " (Tensor)", attributes);
    }

    @NotNull
    public static BufferedImage visualizeGraph(@NotNull IGraph graph) {
        GraphRenderPlan renderPlan = makeRenderPlan(graph);
        return GraphVisualizer.visualizeGraph(renderPlan);
    }
}
