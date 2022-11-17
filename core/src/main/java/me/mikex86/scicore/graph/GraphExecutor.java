package me.mikex86.scicore.graph;

import me.mikex86.scicore.profiling.Profiler;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import org.jetbrains.annotations.NotNull;

import java.util.Deque;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class GraphExecutor {

    // TODO: REVISIT WHETHER INPLACE OPERATIONS CAN BREAK THE BACKWARDS PASS OF OPERATIONS WHICH DEPEND ON THE SAME TENSOR, BUT IN THE PAST, WHEN IT HAD A DIFFERENT VALUE

    public void execute(@NotNull Graph graph) {
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
                        String sectionName = operationGraphNode.getOperationType().name();
                        ITensor output = operationGraphNode.perform();
                        lazyTensor.setResult(output);
                    }
                } else {
                    throw new IllegalStateException("operation graph node has no output.");
                }
            }
        }
    }
}
