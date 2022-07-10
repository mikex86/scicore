package me.mikex86.scicore.op;

import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

public class GraphRecorder implements IGraphRecorder {

    @NotNull
    private final ISciCoreBackend sciCoreBackend;

    @NotNull
    private final Map<Object, Graph.IGraphNode> valueToNodeMap = new IdentityHashMap<>(); // We don't compare based on tensor contents. That's stupid

    @Nullable
    private ITensor lastOutput;

    public GraphRecorder(@NotNull ISciCoreBackend sciCoreBackend) {
        this.sciCoreBackend = sciCoreBackend;
    }

    @Override
    public @NotNull ITensor recordOperation(@NotNull OperationType operationType, @NotNull Object... inputs) {
        IOperation operation = this.sciCoreBackend.getOperation(operationType);
        List<Graph.IGraphNode> inputNodes = new ArrayList<>();
        for (Object input : inputs) {
            Graph.IGraphNode node = valueToNodeMap.get(input);
            if (node == null) {
                if (input instanceof IDerivedTensor) {
                    throw new IllegalStateException("Derive tensor not computed in current graph!");
                } else if (input instanceof ITensor inputTensor) {
                    node = new Graph.TensorDeclarationGraphNode(inputTensor);
                    valueToNodeMap.put(input, node);
                } else {
                    node = new Graph.ValueGraphNode(input);
                    valueToNodeMap.put(input, node);
                }
            }
            inputNodes.add(node);
        }
        ITensor result = sciCoreBackend.lazyOpTensor(operation, List.of(inputs));
        Graph.OperationGraphNode operationGraphNode = new Graph.OperationGraphNode(operationType, inputNodes, result);
        valueToNodeMap.put(result, operationGraphNode);
        lastOutput = result;
        return result;
    }

    @Override
    public @NotNull Graph finish() {
        ITensor lastOutput = this.lastOutput;
        if (lastOutput == null) {
            throw new IllegalStateException("Cannot finish graph recording when not a single operation was recorded");
        }
        Graph.IGraphNode outputNode = valueToNodeMap.get(lastOutput);
        return new Graph(outputNode, sciCoreBackend);
    }

}
