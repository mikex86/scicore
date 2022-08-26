package me.mikex86.scicore.op;

import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import org.jetbrains.annotations.NotNull;

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
        IOperation operation = sciCoreBackend.getOperation(operationType);
        List<Graph.IGraphNode> inputNodes = new ArrayList<>();
        for (ITensor input : inputs) {
            Graph.IGraphNode node = valueToNodeMap.get(input);
            if (node == null) {
                if (input instanceof IDerivedTensor derivedTensor) {
                    input = derivedTensor.result();
                }
                node = new Graph.TensorDeclarationGraphNode(input);
                valueToNodeMap.put(input, node);
            }
            inputNodes.add(node);
        }
        Graph.IOperationContext ctx = new Graph.OperationContext();
        Graph.OperationGraphNode operationGraphNode = new Graph.OperationGraphNode(operationType, inputNodes, ctx);
        ITensor lazyResult = operation.performLazily(ctx, List.of(inputs));
        operationGraphNode.setOutput(lazyResult);
        valueToNodeMap.put(lazyResult, operationGraphNode);
        return lazyResult;
    }

    @Override
    public @NotNull Graph getGraphFor(@NotNull ITensor rootTensor) {
        Graph.IGraphNode rootNode = valueToNodeMap.get(rootTensor);
        if (rootNode == null) {
            throw new IllegalStateException("Tensor was not recorded as an output computed by this graph: " + rootTensor);
        }
        return new Graph(rootNode.deepCopy(), sciCoreBackend);
    }

    @Override
    public void resetRecording() {
        valueToNodeMap.clear();
    }

}
