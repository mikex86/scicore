package me.mikex86.scicore.graph;

import com.google.common.collect.MapMaker;
import me.mikex86.scicore.graph.op.IInplaceOperation;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.backend.OperationRegistry;
import me.mikex86.scicore.graph.op.IOperation;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicBoolean;

public class GraphRecorder implements IGraphRecorder {

    @NotNull
    private final OperationRegistry operationRegistry;

    @NotNull
    private final Map<ITensor, Graph.ITensorNode> valueToNodeMap = new MapMaker()
            .weakKeys()
            .weakValues()
            .makeMap();

    @NotNull
    private final Stack<Map<ITensor, IGraph.ITensorNode>> recordingScopes = new Stack<>();

    {
        recordingScopes.push(new IdentityHashMap<>());
    }

    public GraphRecorder(@NotNull OperationRegistry operationRegistry) {
        this.operationRegistry = operationRegistry;
    }

    @Override
    public @NotNull ITensor recordOperation(@NotNull OperationType operationType, @NotNull OptionBundle optionBundle, @NotNull ITensor... inputs) {
        IOperation operation = operationRegistry.getOperation(operationType);
        List<Graph.IGraphNode> inputNodes = new ArrayList<>();
        if (operation instanceof IInplaceOperation && inputs.length < 1) {
            throw new IllegalStateException("Inplace operation must have at least one input. inputs[0] is the output tensor.");
        }
        ITensor originalInputs0 = inputs[0];
        if (operation instanceof IInplaceOperation && inputs[0] instanceof LazyTensor lazyDst) {
            inputs[0] = lazyDst.lazyCopy(); // See note "lazyCopy" below.
        }
        for (ITensor input : inputs) {
            Graph.ITensorNode node = getGraphNode(input);
            if (node == null) {
                if (input instanceof LazyTensor lazyTensor && !lazyTensor.hasResult()) {
                    throw new IllegalArgumentException("Lost track of tensor");
                }
                node = new Graph.TensorDeclarationGraphNode(input);
                input.setAssociatedGraphNode(node);
                putGraphNode(input, node);
            }
            inputNodes.add(node);
        }
        Graph.IOperationContext ctx = new Graph.OperationContext(optionBundle);
        Graph.OperationGraphNode operationGraphNode = new Graph.OperationGraphNode(operationType, inputNodes, ctx, operationRegistry);

        if (operation instanceof IInplaceOperation) {
            if (originalInputs0 instanceof LazyTensor originalLazyDst) {
                // NOTE 1 - "lazyCopy": We force a forceReevaluation of lazyDst, because the tensor that is being inplace-modified will receive a new associated operation graph node that is this the inplace operation.
                // This ensures that when the tensor is accessed the next time, it does not have a result and the inplace operation is performed.
                // However, if inputs[0] was lazyDst, the result value of inputs[0] before the inplace operation is performed (which is required to perform the operation) would thus be lost,
                // because forceReevaluation() would set the result to null.
                // Therefore, we copy the lazy tensor "lazily" to save this pre-inplace-operation state.
                // This lazy copy contains the result of the pre-inplace-operation lazy tensor, if it was already present,
                // or else the lazy tensor copy is associated with the operation graph node that computed the pre-inplace-operation tensor, allowing the result to be computed lazily up to this point.

                // NOTE 2 - "usafe cleanup": inputs[0] which is a lazy copy of the original inputs[0] is not inside a try-with-resources block, as disposing this tensor would dispose the original inputs[0] as well.
                // The original inputs[0] is still handled by the user, and if by either try-with-resources or GC disposal, the original inputs[0] is disposed, the lazy copy will be disposed as well.
                // This is safe because the lifetime of the lazy copy will never exceed the lifetime of the original inputs[0],
                // as it is either in the same recording scope as the original inputs[0], or in a "further pushed in" recording scope, meaning higher in this.recordingScopes,
                // where the top is the current recording scope, with the shortest lifetime.
                operation.performLazily(ctx, List.of(inputs));
                originalLazyDst.forceReevaluation();

                // NOTE 3 - "setOutput": When multiple inplace-operation nodes are chained, only the last inplace-operation-node should have the output of the tensor that the user handles.
                // This is because the output of the individual inplace operations needs to be a different instance, because otherwise the first operation in the chain will populate the LazyTensor's result field
                // with a premature value, and when the second operation is performed, the result field will be non-null, and the operation will not be performed, as the operation would be considered as "already performed".
                // Therefore, we need different LazyTensor instances for each operation in the chain.
                // We set the output of the graph node of this operation to the original inputs[0], which is the tensor that the user handles and destination for the inplace operation.
                // At this point in time, this operation is the last operation in the chain, so this is the correct output.
                // As we find ourselves in invocations of recordOperation(), where the output of an operation is the input of the next inplace operation,
                // we set the output of these previous operations to new LazyTensor instances to ensure that the GraphExecutor works correctly.
                operationGraphNode.setOutput(originalLazyDst);
                if (originalLazyDst.getAssociatedGraphNode() instanceof Graph.OperationGraphNode prevOperation) {
                    prevOperation.setOutput(new LazyTensor(originalLazyDst.getSciCoreBackend(), originalLazyDst.getShape(), originalLazyDst.getDataType()));
                }
                putGraphNode(originalLazyDst, operationGraphNode);
                return originalLazyDst;
            } else {
                // perform inplace operation eagerly
                ITensor result = operation.perform(ctx, List.of(inputs)); // result == inputs[0] == output tensor
                operationGraphNode.setOutput(result);
                putGraphNode(result, operationGraphNode);
                return result;
            }
        } else {
            ITensor result = operation.performLazily(ctx, List.of(inputs));
            operationGraphNode.setOutput(result);
            putGraphNode(result, operationGraphNode);
            return result;
        }
    }

    @Nullable
    private IGraph.ITensorNode getGraphNode(@NotNull ITensor input) {
        IGraph.ITensorNode node = input.getAssociatedGraphNode(); // this can be null for various reasons
        if (node != null) {
            return node;
        }
        return valueToNodeMap.get(input);
    }

    public void putGraphNode(@NotNull ITensor tensor, @NotNull IGraph.ITensorNode graphNode) {
        tensor.setAssociatedGraphNode(graphNode);
        this.valueToNodeMap.put(tensor, graphNode);
        this.recordingScopes.peek().put(tensor, graphNode);
    }

    @Override
    public @NotNull Graph getExecutionGraphTo(@NotNull ISciCoreBackend backend, @NotNull ITensor rootTensor) {
        Graph.IGraphNode rootNode = getGraphNode(rootTensor);
        if (rootNode == null) {
            throw new IllegalArgumentException("Tensor was not recorded as an output computed by this graph");
        }

        // Creates a graph from the recorded nodes
        // Operation Nodes, whose output is already computed, will be present as tensor declaration nodes, as opposed to operation nodes.
        Queue<IGraph.IGraphNode> toVisit = new ArrayDeque<>();
        toVisit.add(rootNode);

        // Reverse topological reorder until we reach already computed nodes
        // Note that the same node can be reached from multiple paths, at different depths.
        // When we add a node to this list that was already present, we need to remove the previous occurrence and add it again at the front.
        // This way, the first occurrence of a node in the list will be the one with the highest depth (meaning executed first).
        // This is important because we want to compute the nodes in the order of their depth, so that we can compute the inputs before the outputs.
        Deque<Graph.OperationGraphNode> reverseTopology = new LinkedList<>();
        while (!toVisit.isEmpty()) {
            IGraph.IGraphNode node = toVisit.remove();
            if (node instanceof Graph.OperationGraphNode operationNode) {
                List<IGraph.IGraphNode> inputs = operationNode.getInputs();
                for (IGraph.IGraphNode input : inputs) {
                    if (input instanceof Graph.OperationGraphNode inputOperationNode) {
                        if (!inputOperationNode.hasOutput() || (inputOperationNode.getOutput() instanceof LazyTensor lazyTensor && !lazyTensor.hasResult())) {
                            toVisit.add(input);
                        }
                    } else {
                        toVisit.add(input);
                    }
                }
                reverseTopology.remove(operationNode); // remove if already present
                reverseTopology.addFirst(operationNode);
            }
        }

        // Maps from the original node to a copy of the list of inputs of that node.
        Map<Graph.OperationGraphNode, List<IGraph.IGraphNode>> nodeToInputs = new HashMap<>();

        Map<Graph.IGraphNode, Graph.IGraphNode> nodeToCopy = new IdentityHashMap<>();

        // iterate in reverse topological order and create copies via #copyNode.
        // Iterating in reverse topological order ensures that leaf nodes are visited first, and thus their copies are created first.
        // This is important because nodes that depend on these leaf nodes will need to reference the copies of the leaf nodes,
        // which means their copies must already be created at that point in time.
        for (Graph.OperationGraphNode operationNode : reverseTopology) {
            List<IGraph.IGraphNode> inputs = operationNode.getInputs();
            List<IGraph.IGraphNode> copies = new ArrayList<>(inputs.size());
            for (IGraph.IGraphNode input : inputs) {
                IGraph.IGraphNode alreadyExistingCopy = nodeToCopy.get(input);
                if (alreadyExistingCopy == null) {
                    IGraph.IGraphNode copy = copyNodeWithAlreadyComputedAsConstants(input, nodeToInputs);
                    copies.add(copy);
                    nodeToCopy.put(input, copy);
                } else {
                    copies.add(alreadyExistingCopy);
                }
            }
            nodeToInputs.put(operationNode, copies);
        }

        IGraph.IGraphNode rootCopy = copyNodeWithAlreadyComputedAsConstants(rootNode, nodeToInputs);
        return new Graph(rootCopy, backend, operationRegistry);
    }

    /**
     * Copies a node and makes the outputs of already computed operations constant tensor nodes.
     *
     * @param node         The node to copy
     * @param nodeToInputs A map from the original node to a copy of the list of inputs of that node. Populated in reverse topological order, thus the inputs for a downstream node will exist by the time we get there.
     * @return The copy of the node
     */
    @NotNull
    private IGraph.IGraphNode copyNodeWithAlreadyComputedAsConstants(@NotNull IGraph.IGraphNode node, @NotNull Map<Graph.OperationGraphNode, List<IGraph.IGraphNode>> nodeToInputs) {
        if (node instanceof Graph.TensorDeclarationGraphNode tensorDeclarationGraphNode) {
            return new Graph.TensorDeclarationGraphNode(tensorDeclarationGraphNode.getValue());
        } else if (node instanceof Graph.OperationGraphNode operationGraphNode) {
            if (!operationGraphNode.hasOutput() || (operationGraphNode.getOutput() instanceof LazyTensor lazyTensor && !lazyTensor.hasResult())) {
                List<IGraph.IGraphNode> inputCopies = nodeToInputs.get(operationGraphNode);
                Graph.OperationGraphNode operationNodeCpy = new Graph.OperationGraphNode(operationGraphNode.getOperationType(), inputCopies, operationGraphNode.getOperationContext(), operationRegistry);
                if (operationGraphNode.hasOutput()) {
                    operationNodeCpy.setOutput(operationGraphNode.getOutput());
                }
                return operationNodeCpy;
            } else {
                // If the input node has already been computed, the node will be assumed constant and thus will be represented as a tensor declaration node.
                return new Graph.TensorDeclarationGraphNode(operationGraphNode.getOutput());
            }
        } else {
            throw new IllegalArgumentException("Unknown node type: " + node.getClass().getName());
        }
    }

    @Override
    public @NotNull Graph getBackpropagationGraphTo(@NotNull ISciCoreBackend backend, @NotNull ITensor rootTensor, @NotNull List<ITensor> parameters) {
        Graph.IGraphNode rootNode = getGraphNode(rootTensor);
        if (rootNode == null) {
            throw new IllegalArgumentException("Tensor was not recorded as an output computed by this graph");
        }

        // Creates a graph from the recorded nodes
        // Operation Nodes, whose output is already computed, will be present as tensor declaration nodes, as opposed to operation nodes.
        Queue<IGraph.IGraphNode> toVisit = new ArrayDeque<>();
        toVisit.add(rootNode);

        // Reverse topological reorder until we reach already computed nodes
        // Note that the same node can be reached from multiple paths, at different depths.
        // When we add a node to this list that was already present, we need to remove the previous occurrence and add it again at the front.
        // This way, the first occurrence of a node in the list will be the one with the highest depth (meaning executed first).
        // This is important because we want to compute the nodes in the order of their depth, so that we can compute the inputs before the outputs.
        Deque<Graph.OperationGraphNode> reverseTopology = new LinkedList<>();
        while (!toVisit.isEmpty()) {
            IGraph.IGraphNode node = toVisit.remove();
            if (node instanceof Graph.OperationGraphNode operationNode) {
                List<IGraph.IGraphNode> inputs = operationNode.getInputs();
                toVisit.addAll(inputs);
                reverseTopology.remove(operationNode); // remove if already present
                reverseTopology.addFirst(operationNode);
            }
        }

        // Maps from the original node to a copy of the list of inputs of that node.
        Map<Graph.OperationGraphNode, List<IGraph.IGraphNode>> nodeToInputs = new HashMap<>();

        Map<Graph.IGraphNode, Graph.IGraphNode> nodeToCopy = new IdentityHashMap<>();

        // iterate in reverse topological order and create copies via #copyNode.
        // Iterating in reverse topological order ensures that leaf nodes are visited first, and thus their copies are created first.
        // This is important because nodes that depend on these leaf nodes will need to reference the copies of the leaf nodes,
        // which means their copies must already be created at that point in time.
        for (Graph.OperationGraphNode operationNode : reverseTopology) {
            List<IGraph.IGraphNode> inputs = operationNode.getInputs();
            List<IGraph.IGraphNode> copies = new ArrayList<>(inputs.size());
            for (IGraph.IGraphNode input : inputs) {
                IGraph.IGraphNode alreadyExistingCopy = nodeToCopy.get(input);
                if (alreadyExistingCopy == null) {
                    IGraph.IGraphNode copy = copyNodeIncludeAlreadyComputed(input, nodeToInputs);
                    if (copy == null) {
                        throw new IllegalStateException("Cannot construct backpropagation graph, as the graph leads to tensors that are already disposed. Cannot safely continue. " +
                                                        "Did the optimization operations leak into your main graph? " +
                                                        "Check your usage of GraphRecorder#recordWithScope() or GraphRecorder#scopedRecording() in kotlin, or GraphRecorder#resetRecording()");
                    }
                    copies.add(copy);
                    nodeToCopy.put(input, copy);
                } else {
                    copies.add(alreadyExistingCopy);
                }
            }
            nodeToInputs.put(operationNode, copies);
        }

        IGraph.IGraphNode rootCopy = copyNodeIncludeAlreadyComputed(rootNode, nodeToInputs);
        if (rootCopy == null) {
            throw new IllegalArgumentException("Cannot construct backpropagation graph to a tensor that was already disposed");
        }
        Graph graph = new Graph(rootCopy, backend, operationRegistry);
        graph.requestGradientsFor(parameters);
        return graph;
    }

    private int nBytesProbablyDeletedSinceLastAsyncGC = 0;
    private int nBytesProbablyDeletedSinceLastOnSameThreadGC = 0;

    @NotNull
    private final AtomicBoolean shouldRunGC = new AtomicBoolean(false);

    @NotNull
    private final Thread gcThread = new Thread(() -> {
        while (true) {
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            if (shouldRunGC.getAndSet(false)) {
                System.gc();
            }
        }
    }, "GC-Invoker-Thread");

    {
        gcThread.setDaemon(true);
        gcThread.start();
    }

    private void checkShouldRunGC() {
        if (nBytesProbablyDeletedSinceLastAsyncGC > 200_000_000) { // 200 Mb
            shouldRunGC.set(true);
            nBytesProbablyDeletedSinceLastAsyncGC = 0;
        }
        if (nBytesProbablyDeletedSinceLastOnSameThreadGC > 2_000_000_000) { // 2 GB
            System.gc();
            nBytesProbablyDeletedSinceLastOnSameThreadGC = 0;
        }
    }

    @Override
    public void resetRecording() {
        this.valueToNodeMap.clear();
        while (!this.recordingScopes.isEmpty()) {
            Map<ITensor, IGraph.ITensorNode> scope = this.recordingScopes.pop();
            for (Map.Entry<ITensor, IGraph.ITensorNode> entry : scope.entrySet()) {
                disposeIfPossible(entry);
            }
            checkShouldRunGC();
        }
        this.recordingScopes.push(new IdentityHashMap<>());
        checkShouldRunGC();
    }

    @Override
    public void recordWithScope(@NotNull Callable<Void> recording) {
        this.recordingScopes.push(new IdentityHashMap<>());
        try {
            recording.call();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        Map<ITensor, IGraph.ITensorNode> scope = this.recordingScopes.pop();
        for (Map.Entry<ITensor, IGraph.ITensorNode> entry : scope.entrySet()) {
            disposeIfPossible(entry);
        }
        checkShouldRunGC();
    }

    private void disposeIfPossible(@NotNull Map.Entry<ITensor, IGraph.ITensorNode> entry) {
        ITensor value = entry.getKey();
        IGraph.ITensorNode node = entry.getValue();

        if (node instanceof Graph.OperationGraphNode operationGraphNode) {
            // dispose operation context
            Graph.IOperationContext ctx = operationGraphNode.getOperationContext();
            Map<String, ITensor> savedTensors = ctx.getSavedTensors();
            OptionBundle optionBundle = ctx.getOptionBundle();
            // Get the node value
            ITensor nodeValue = node.getValue();
            while (nodeValue instanceof LazyTensor lazyTensor) {
                if (lazyTensor.hasResult()) {
                    nodeValue = lazyTensor.result();
                } else {
                    nodeValue = null;
                }
            }
            for (ITensor savedTensor : savedTensors.values()) {
                // We only dispose the saved tensors if the same tensor instance is not also the output of the operation.
                // if the node value is null (meaning the lazy result was not yet computed), we can't know if the saved tensor is the same as the node value, so we dispose it anyway.
                // We can do that, because we are at the end of the scope. Here, if the lazy result is not yet computed, it will never be computed, so we can safely dispose the saved tensors.
                if (nodeValue == null || savedTensor != nodeValue) {
                    savedTensor.dispose();
                }
            }
            savedTensors.clear();
            optionBundle.dispose();
        }
        if ((!(node instanceof Graph.OperationGraphNode operationGraphNode) || !operationGraphNode.getOperationType().isInplace())) { // Inplace operations do not own their output tensor, so we don't dispose it.
            value.setAssociatedGraphNode(null);
            if (value.isDeReferenced()) {
                if (!value.isDisposed()) {
                    value.dispose();
                }
            } else {
                nBytesProbablyDeletedSinceLastAsyncGC += value.getNumBytes();
                nBytesProbablyDeletedSinceLastOnSameThreadGC += value.getNumBytes();
                node.deleteValue();
            }
            this.valueToNodeMap.remove(value);
        }
    }


    /**
     * Copies a node without making already computed nodes constant tensor nodes.
     *
     * @param node         The node to copy
     * @param nodeToInputs A map from the original node to a copy of the list of inputs of that node. Populated in reverse topological order, thus the inputs for a downstream node will exist by the time we get there.
     * @return The copy of the node, or null, if some inputs of the node are already disposed.
     */
    @Nullable
    private IGraph.IGraphNode copyNodeIncludeAlreadyComputed(@NotNull IGraph.IGraphNode node, @NotNull Map<Graph.OperationGraphNode, List<IGraph.IGraphNode>> nodeToInputs) {
        if (node instanceof Graph.TensorDeclarationGraphNode tensorDeclarationGraphNode) {
            if (!tensorDeclarationGraphNode.hasValue()) {
                return null;
            }
            return new Graph.TensorDeclarationGraphNode(tensorDeclarationGraphNode.getValue());
        } else if (node instanceof Graph.OperationGraphNode operationGraphNode) {
            List<IGraph.IGraphNode> inputCopies = nodeToInputs.get(operationGraphNode);
            if (inputCopies == null) {
                // We can get here if the node is a parameter node, which is an operation node itself (eg. because it was already updated once by the optimizer),
                // but, at this point we don't want to let the graph span into the previous training step because what the fuck.
                // So, here we also represent the parameter as a constant and thus as a tensor declaration node.
                if (!operationGraphNode.hasOutput()) {
                    throw new IllegalStateException("Operation leaf node with omitted inputs without already computed outputs encountered. This should not happen.");
                }
                return new Graph.TensorDeclarationGraphNode(operationGraphNode.getOutput());
            }
            Graph.OperationGraphNode operationNodeCpy = new Graph.OperationGraphNode(operationGraphNode.getOperationType(), inputCopies, operationGraphNode.getOperationContext(), operationRegistry);
            if (operationGraphNode.hasOutput()) {
                operationNodeCpy.setOutput(operationGraphNode.getOutput());
            }
            return operationNodeCpy;
        } else {
            throw new IllegalArgumentException("Unknown node type: " + node.getClass().getName());
        }
    }

}
