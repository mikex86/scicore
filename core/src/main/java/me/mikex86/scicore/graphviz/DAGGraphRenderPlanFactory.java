package me.mikex86.scicore.graphviz;

import me.mikex86.matplotlib.graphviz.GraphRenderPlan;
import me.mikex86.matplotlib.graphviz.GraphVisualizer;
import me.mikex86.scicore.ITensor;
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
        int rowIndex = 0;

        IGraph.IGraphNode currentNode = graph.getOutputNode();

        Map<IGraph.IGraphNode, GraphRenderPlan.IGraphNode> graphNodes = new HashMap<>();

        // add current node
        {
            GraphRenderPlan.Column column = new GraphRenderPlan.Column(List.of(toGraphNode(graphNodes, currentNode)));
            rowStack.getRow(rowIndex).add(column);
        }

        makeRenderPlan(graphNodes, rowStack, currentNode, rowIndex + 1);
        return new GraphRenderPlan(rowStack);
    }

    @NotNull
    private static GraphRenderPlan.IGraphNode toGraphNode(@NotNull Map<IGraph.IGraphNode, GraphRenderPlan.IGraphNode> graphNodes, @NotNull IGraph.IGraphNode dagNode) {
        GraphRenderPlan.IGraphNode existingDagNode = graphNodes.get(dagNode);
        if (existingDagNode != null) {
            return existingDagNode;
        }
        if (dagNode instanceof Graph.TensorDeclarationGraphNode tensorNode) {
            ITensor value = tensorNode.getValue();
            Map<String, String> attributes = new LinkedHashMap<>();
            attributes.put("dataType", value.getDataType().toString());
            attributes.put("shape", ShapeUtils.toString(value.getShape()));
            attributes.put("isScalar", Boolean.toString(value.isScalar()));
            GraphRenderPlan.IGraphNode.DataNode vizNode = new GraphRenderPlan.IGraphNode.DataNode(tensorNode.getName() + " (Tensor)", attributes);
            graphNodes.put(tensorNode, vizNode);
            return vizNode;
        } else if (dagNode instanceof Graph.OperationGraphNode operation) {
            List<IGraph.@NotNull IGraphNode> inputs = operation.getInputs();
            GraphRenderPlan.IGraphNode.Interconnect vizNode = new GraphRenderPlan.IGraphNode.Interconnect(operation.getName(), inputs.stream().map(dagNode1 -> toGraphNode(graphNodes, dagNode1)).toList());
            graphNodes.put(operation, vizNode);
            return vizNode;
        } else {
            throw new IllegalArgumentException("Unknown DAG node type: " + dagNode.getClass());
        }
    }

    private static void makeRenderPlan(@NotNull Map<IGraph.IGraphNode, GraphRenderPlan.IGraphNode> graphNodes, @NotNull GraphRenderPlan.RowStack rowStack, @NotNull IGraph.IGraphNode currentNode, int rowIndex) {
        if (currentNode instanceof Graph.OperationGraphNode operationNode) {
            makeOperationNodeRenderPlan(graphNodes, rowStack, operationNode, rowIndex);
        }
    }

    private static void makeOperationNodeRenderPlan(@NotNull Map<IGraph.IGraphNode, GraphRenderPlan.IGraphNode> graphNodes, @NotNull GraphRenderPlan.RowStack rowStack, @NotNull Graph.OperationGraphNode currentNode, int rowIndex) {
        // add inputs
        {
            List<IGraph.IGraphNode> inputs = currentNode.getInputs();

            // Cluster the input nodes for the current node as a column in the current row
            {
                GraphRenderPlan.Column column = new GraphRenderPlan.Column(inputs.stream().map(dagNode -> toGraphNode(graphNodes, dagNode)).toList());
                rowStack.getRow(rowIndex).add(column);
            }

            // Recursive render planning for inputs. The inputs nodes themselves are already in the plan,
            // now we just need recursively add the inputs of the inputs in the next row.
            // Note this recursion is in a loop. The recursion will back off to rowIndex + 1
            // potentially multiple times if inputs.size() > 0. This ensures that multiple columns are placed in the
            // same row, such that their row is equal to the node's graph depth in the graph.
            for (IGraph.IGraphNode input : inputs) {
                makeRenderPlan(graphNodes, rowStack, input, rowIndex + 1);
            }
        }
    }

    @NotNull
    public static BufferedImage visualizeGraph(@NotNull IGraph graph) {
        GraphRenderPlan renderPlan = makeRenderPlan(graph);
        return GraphVisualizer.visualizeGraph(renderPlan);
    }
}
