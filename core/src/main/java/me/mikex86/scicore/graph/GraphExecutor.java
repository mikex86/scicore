package me.mikex86.scicore.graph;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import org.jetbrains.annotations.NotNull;

import java.util.Deque;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class GraphExecutor {

    private static int numOperations = 0;

    public void execute(@NotNull ISciCoreBackend backend, @NotNull Graph graph) {
        IGraph.ITensorNode outputNode = (IGraph.ITensorNode) graph.getOutputNode();
        Queue<IGraph.IGraphNode> toVisit = new LinkedList<>();
        toVisit.add(outputNode);

        Deque<IGraph.IGraphNode> reverseTopologicalOrder = new LinkedList<>();
        while (!toVisit.isEmpty()) {
            IGraph.IGraphNode node = toVisit.remove();
            reverseTopologicalOrder.addFirst(node);
            if (node instanceof Graph.OperationGraphNode operationGraphNode) {
                List<IGraph.IGraphNode> inputs = operationGraphNode.getInputs();
                for (IGraph.IGraphNode input : inputs) {
                    if (input instanceof Graph.OperationGraphNode inputOperationGraphNode) {
                        toVisit.add(inputOperationGraphNode);
                    }
                }
            }
        }

        for (IGraph.IGraphNode graphNode : reverseTopologicalOrder) {
            if (graphNode instanceof Graph.OperationGraphNode operationGraphNode) {
                if (operationGraphNode.hasOutput()) {
                    ITensor nodeOutput = operationGraphNode.getOutput();
                    if (nodeOutput instanceof LazyTensor lazyTensor && !lazyTensor.hasResult()) {
                        ITensor output = operationGraphNode.perform();
                        lazyTensor.setResult(output);
                        numOperations++;
                    }
                } else {
                    throw new IllegalStateException("operation graph node has no output.");
                }
            }
        }
        backend.synchronize();
    }

    public static int getNumOperations() {
        return numOperations;
    }
}
