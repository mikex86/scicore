package me.mikex86.scicore.op;

import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.*;

public class GraphRecorder implements IGraphRecorder {

    @NotNull
    private final ISciCoreBackend sciCoreBackend;

    @NotNull
    private final WeakHashMap<Object, Graph.IGraphNode> valueToNodeMap = new WeakHashMap<>(); // We don't compare based on tensor contents. That's stupid

    public GraphRecorder(@NotNull ISciCoreBackend sciCoreBackend) {
        this.sciCoreBackend = sciCoreBackend;
    }

    @Override
    public @NotNull ITensor recordOperation(@NotNull OperationType operationType, @NotNull ITensor... inputs) {
        IOperation operation = this.sciCoreBackend.getOperation(operationType);
        List<Graph.IGraphNode> inputNodes = new ArrayList<>();
        for (ITensor input : inputs) {
            Graph.IGraphNode node = valueToNodeMap.get(input);
            if (node == null) {
                if (input instanceof IDerivedTensor) {
                    throw new IllegalStateException("Derive tensor not computed in current graph!");
                } else {
                    node = new Graph.TensorDeclarationGraphNode(input);
                    valueToNodeMap.put(input, node);
                }
            }
            inputNodes.add(node);
        }
        ITensor result = sciCoreBackend.lazyOpTensor(operation, List.of(inputs));
        Graph.OperationGraphNode operationGraphNode = new Graph.OperationGraphNode(operationType, inputNodes, result);
        valueToNodeMap.put(result, operationGraphNode);
        return result;
    }

    @Override
    public @NotNull Graph getGraphFor(@NotNull ITensor rootTensor) {
        Graph.IGraphNode rootNode = valueToNodeMap.get(rootTensor);
        if (rootNode == null) {
            throw new IllegalStateException("Tensor was not recorded as an output computed by this graph: " + rootTensor);
        }
        return new Graph(rootNode.deepCopy(), sciCoreBackend);
    }

}
