package me.mikex86.scicore.graph;

import com.google.common.collect.MapMaker;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.backend.OperationRegistry;
import me.mikex86.scicore.graph.op.IOperation;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.*;

public class GraphRecorder implements IGraphRecorder {

    @NotNull
    private final OperationRegistry operationRegistry;

    @NotNull
    private final Map<ITensor, Graph.IGraphNode> valueToNodeMap = new MapMaker()
            .weakKeys()
            .weakValues()
            .makeMap();

    /**
     * Maps graph nodes to the list of graph nodes that use them as inputs.
     */
    @NotNull
    private final Map<IGraph.IGraphNode, List<Graph.OperationGraphNode>> nodeToUsages = new MapMaker()
            .weakKeys()
            .makeMap();

    public GraphRecorder(@NotNull OperationRegistry operationRegistry) {
        this.operationRegistry = operationRegistry;
    }

    @Override
    public @NotNull ITensor recordOperation(@NotNull OperationType operationType, @NotNull OptionBundle optionBundle, @NotNull ITensor... inputs) {
        IOperation operation = operationRegistry.getOperation(operationType);
        List<Graph.IGraphNode> inputNodes = new ArrayList<>();
        for (ITensor input : inputs) {
            Graph.IGraphNode node = getGraphNode(input);
            if (node == null) {
                if (input instanceof LazyTensor lazyTensor && !lazyTensor.hasResult()) {
                    throw new IllegalArgumentException("Lost track of tensor");
                }
                node = new Graph.TensorDeclarationGraphNode(input);
                input.setReferenceToAssociatedGraphNode(node);
                putGraphNode(input, node);
            }
            inputNodes.add(node);
        }
        Graph.IOperationContext ctx = new Graph.OperationContext(optionBundle);
        Graph.OperationGraphNode operationGraphNode = new Graph.OperationGraphNode(operationType, inputNodes, ctx, operationRegistry);
        ITensor result = operation.performLazily(ctx, List.of(inputs));
//        if (operation instanceof IInplaceOperation) {
//            operationGraphNode.setOutput(new LazyTensor(result.getSciCoreBackend(), result.getShape(), result.getDataType()));
//        } else {
        operationGraphNode.setOutput(result);
//        }
        putGraphNode(result, operationGraphNode);
        return result;
    }

    @Nullable
    private IGraph.IGraphNode getGraphNode(@NotNull ITensor input) {
        return this.valueToNodeMap.get(input);
    }

    private boolean graphNodeExistsFor(@NotNull ITensor input) {
        return this.valueToNodeMap.containsKey(input);
    }

    public void putGraphNode(@NotNull ITensor tensor, @NotNull IGraph.IGraphNode graphNode) {
        if (graphNode instanceof Graph.OperationGraphNode operationGraphNode) {
            for (IGraph.IGraphNode input : operationGraphNode.getInputs()) {
                nodeToUsages.computeIfAbsent(input, k -> new ArrayList<>()).add(operationGraphNode);
            }
        }
        tensor.setReferenceToAssociatedGraphNode(graphNode);
        this.valueToNodeMap.put(tensor, graphNode);
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
                    copies.add(copy);
                    nodeToCopy.put(input, copy);
                } else {
                    copies.add(alreadyExistingCopy);
                }
            }
            nodeToInputs.put(operationNode, copies);
        }

        IGraph.IGraphNode rootCopy = copyNodeIncludeAlreadyComputed(rootNode, nodeToInputs);
        Graph graph = new Graph(rootCopy, backend, operationRegistry);
        graph.requestGradientsFor(parameters);
        return graph;
    }

    private int nBytesDeletedSinceLastGC = 0;

    @Override
    public void dropHistory(@NotNull ITensor tensor) {
        IGraph.IGraphNode node = getGraphNode(tensor);
        if (node == null) {
            throw new IllegalArgumentException("Tensor was not recorded as an output computed by this graph");
        }

        if (tensor instanceof LazyTensor lazyTensor) {
            lazyTensor.result();
        }

        // TODO: FIX SOME NODES NOT BEING CLEANED
        if (node instanceof Graph.OperationGraphNode operationNode) {

            // Replace the node that computed the tensor with a tensor declaration node that holds a constant tensor with the same value.
            {
                Graph.TensorDeclarationGraphNode constantReplacement = new Graph.TensorDeclarationGraphNode(tensor);
                List<Graph.OperationGraphNode> usages = nodeToUsages.get(operationNode);
                if (usages != null) {
                    for (Graph.OperationGraphNode usage : usages) {
                        usage.replaceInputs(node, constantReplacement);
                    }
                }
                putGraphNode(tensor, constantReplacement);
            }

            // Nullify tensors of all nodes that are now detached from the graph.
            Queue<IGraph.IGraphNode> toVisit = new LinkedList<>();
            toVisit.add(operationNode);

            Set<IGraph.IGraphNode> deleted = new HashSet<>();

            while (!toVisit.isEmpty()) {
                IGraph.IGraphNode currentNode = toVisit.remove();
                if (currentNode instanceof Graph.OperationGraphNode operationGraphNode) {
                    toVisit.addAll(operationGraphNode.getInputs());
                }
                if (currentNode instanceof IGraph.ITensorNode tensorNode) {
                    List<Graph.OperationGraphNode> usages = nodeToUsages.get(tensorNode);
                    if (usages == null || usages.isEmpty()) {
                        nodeToUsages.remove(tensorNode);

                        if (tensorNode.hasValue()) {
                            ITensor value = tensorNode.getValue();
                            valueToNodeMap.remove(value);
                            tensorNode.deleteValue();
                            deleted.add(tensorNode);
                            nBytesDeletedSinceLastGC += value.getNumBytes();
                            value = null; // help GC
                        }
                    } else {
                        Iterator<Graph.OperationGraphNode> iterator = usages.iterator();
                        while (iterator.hasNext()) {
                            Graph.OperationGraphNode usage = iterator.next();
                            if (deleted.contains(usage)) {
                                iterator.remove();
                            }
                        }
                        if (usages.isEmpty()) {
                            nodeToUsages.remove(tensorNode);
                            if (tensorNode.hasValue()) {
                                ITensor value = tensorNode.getValue();
                                valueToNodeMap.remove(value);
                                tensorNode.deleteValue();
                                deleted.add(tensorNode);
                                nBytesDeletedSinceLastGC += value.getNumBytes();
                                value = null; // help GC
                            }
                        }
                    }
                }
            }
        }
        if (nBytesDeletedSinceLastGC > 500_000_000) { // 100 Mb
            System.gc();
            nBytesDeletedSinceLastGC = 0;
        }
    }


    /**
     * Copies a node without making already computed nodes constant tensor nodes.
     *
     * @param node         The node to copy
     * @param nodeToInputs A map from the original node to a copy of the list of inputs of that node. Populated in reverse topological order, thus the inputs for a downstream node will exist by the time we get there.
     * @return The copy of the node
     */
    @NotNull
    private IGraph.IGraphNode copyNodeIncludeAlreadyComputed(@NotNull IGraph.IGraphNode node, @NotNull Map<Graph.OperationGraphNode, List<IGraph.IGraphNode>> nodeToInputs) {
        if (node instanceof Graph.TensorDeclarationGraphNode tensorDeclarationGraphNode) {
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
