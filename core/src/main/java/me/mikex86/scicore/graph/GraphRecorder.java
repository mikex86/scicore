package me.mikex86.scicore.graph;

import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.backend.OperationRegistry;
import me.mikex86.scicore.graph.op.IInplaceOperation;
import me.mikex86.scicore.graph.op.IOperation;
import org.jetbrains.annotations.NotNull;

import java.util.*;

public class GraphRecorder implements IGraphRecorder {

    @NotNull
    private final OperationRegistry operationRegistry;

    @NotNull
    private final IdentityHashMap<ITensor, Graph.IGraphNode> valueToNodeMap = new IdentityHashMap<>(); // We don't compare based on tensor contents. That's stupid

    public GraphRecorder(@NotNull OperationRegistry operationRegistry) {
        this.operationRegistry = operationRegistry;
    }

    @Override
    public @NotNull ITensor recordOperation(@NotNull OperationType operationType, @NotNull OptionBundle optionBundle, @NotNull ITensor... inputs) {
        IOperation operation = operationRegistry.getOperation(operationType);

        List<Graph.IGraphNode> inputNodes = new ArrayList<>();
        for (ITensor input : inputs) {
            Graph.IGraphNode node = valueToNodeMap.get(input);
            if (node == null) {
                node = new Graph.TensorDeclarationGraphNode(input);
                valueToNodeMap.put(input, node);
            }
            inputNodes.add(node);
        }
        Graph.IOperationContext ctx = new Graph.OperationContext(optionBundle);
        Graph.OperationGraphNode operationGraphNode = new Graph.OperationGraphNode(operationType, inputNodes, ctx, operationRegistry);
        ITensor result = operation.performLazily(ctx, List.of(inputs));
        if (operation instanceof IInplaceOperation) {
            operationGraphNode.setOutput(new LazyTensor(result.getSciCoreBackend(), result.getShape(), result.getDataType())); // force re-evaluation
        } else {
            operationGraphNode.setOutput(result);
        }
        valueToNodeMap.put(result, operationGraphNode);
        return result;
    }

    @Override
    public @NotNull Graph getExecutionGraphTo(@NotNull ISciCoreBackend backend, @NotNull ITensor rootTensor) {
        Graph.IGraphNode rootNode = valueToNodeMap.get(rootTensor);
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
    public @NotNull Graph getBackpropagationGraphTo(@NotNull ISciCoreBackend sciCoreBackend, @NotNull ITensor rootTensor, @NotNull List<ITensor> parameters) {
        Graph.IGraphNode rootNode = valueToNodeMap.get(rootTensor);
        if (rootNode == null) {
            throw new IllegalArgumentException("Tensor was not recorded as an output computed by this graph");
        }

        for (ITensor parameter : parameters) {
            if (!valueToNodeMap.containsKey(parameter)) {
                throw new IllegalArgumentException("Parameter was not recorded as a value of this graph");
            }
        }

        Map<ITensor, IGraph.IGraphNode> parameterToNodeMap = new IdentityHashMap<>();
        Map<ITensor, Set<IGraph.IGraphNode>> visitedWhenSearchingForNode = new IdentityHashMap<>();

        for (ITensor parameter : parameters) {
            visitedWhenSearchingForNode.put(parameter, Collections.newSetFromMap(new IdentityHashMap<>()));
        }

        // Depth first search is dangerous here, because we will bias towards traversing back in time
        // We assume that the parameter is not that far back in execution time.
        // Figuratively speaking, we want to exhaust the present, before going back in time.
        // Thus, we use a breadth first search to find the parameter.
        // We keep track of the nodes we visited to find each parameter.
        // This is to make the depth first search that follows afterwards more efficient.
        {
            for (ITensor parameter : parameters) {
                Queue<IGraph.IGraphNode> toVisit = new LinkedList<>();
                toVisit.add(rootNode);

                Set<IGraph.IGraphNode> visited = Collections.newSetFromMap(new IdentityHashMap<>());
                while (!toVisit.isEmpty()) {
                    IGraph.IGraphNode node = toVisit.remove();
                    if (visited.contains(node)) {
                        continue;
                    }
                    visited.add(node);
                    visitedWhenSearchingForNode.get(parameter).add(node);
                    if (node instanceof IGraph.ITensorNode tensorNode) {
                        if (tensorNode.getValue().isSame(parameter)) {
                            parameterToNodeMap.put(parameter, node);
                            break;
                        }
                    }
                    if (node instanceof Graph.OperationGraphNode operationNode) {
                        toVisit.addAll(operationNode.getInputs());
                    }
                }
            }
        }

        for (ITensor parameter : parameters) {
            if (!parameterToNodeMap.containsKey(parameter)) {
                throw new IllegalArgumentException("Parameter was not found in the graph");
            }
        }

        // We need this depth first search because we need to find the path from the parameter to the root node.
        // The set of nodes the breath first search visited will make the search more efficient
        // Build up parameterToPathMap
        Map<ITensor, Stack<IGraph.IGraphNode>> parameterToPathMap = getPathsToParameters(rootNode, parameters, visitedWhenSearchingForNode);

        // Build serialized reverse topology
        Queue<IGraph.IGraphNode> reverseTopology = buildReverseTopology(parameterToPathMap);

        IGraph.IGraphNode rootCopy = copyWithDependencies(rootNode, reverseTopology, parameterToNodeMap);
        Graph graph = new Graph(rootCopy, sciCoreBackend, operationRegistry);
        graph.requestGradientsFor(parameters);
        return graph;
    }

    @NotNull
    private Queue<IGraph.IGraphNode> buildReverseTopology(@NotNull Map<ITensor, Stack<IGraph.IGraphNode>> parameterToPathMap) {
        Queue<IGraph.IGraphNode> reverseTopology = new LinkedList<>();
        for (Stack<IGraph.IGraphNode> pathToParameter : parameterToPathMap.values()) {
            // add path to parameter reversed
            while (!pathToParameter.isEmpty()) {
                IGraph.IGraphNode pathNode = pathToParameter.pop();
                reverseTopology.remove(pathNode); // remove old occurrence, if any
                // re-add to the end. Re-adding to the end ensures that the nodes will not come before all their inputs in the list
                reverseTopology.add(pathNode);
            }
        }
        return reverseTopology;
    }

    @NotNull
    private IGraph.IGraphNode copyWithDependencies(@NotNull IGraph.IGraphNode rootNode, @NotNull Iterable<IGraph.IGraphNode> reverseTopology, @NotNull Map<ITensor, IGraph.IGraphNode> parameterToNodeMap) {
        // Maps from the original node to a copy of the list of inputs of that node.
        Map<Graph.OperationGraphNode, List<IGraph.IGraphNode>> nodeToInputs = new HashMap<>();
        Map<Graph.IGraphNode, Graph.IGraphNode> nodeToCopy = new IdentityHashMap<>();
        for (IGraph.IGraphNode node : reverseTopology) {
            if (parameterToNodeMap.containsValue(node)) {
                continue;
            }
            if (node instanceof Graph.OperationGraphNode operationNode) {
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
        }
        return copyNodeIncludeAlreadyComputed(rootNode, nodeToInputs);
    }

    @NotNull
    private Map<ITensor, Stack<IGraph.IGraphNode>> getPathsToParameters(@NotNull IGraph.IGraphNode rootNode, @NotNull List<ITensor> parameters, @NotNull Map<ITensor, Set<IGraph.IGraphNode>> visitedWhenSearchingForNode) {
        Map<ITensor, Stack<IGraph.IGraphNode>> parameterToPathMap = new IdentityHashMap<>();

        // Visit nodes of the graph until we have reached all parameters
        Set<ITensor> parametersToVisit = Collections.newSetFromMap(new IdentityHashMap<>());
        parametersToVisit.addAll(parameters);

        Deque<IGraph.IGraphNode> toVisit = new LinkedList<>();
        toVisit.addFirst(rootNode);

        for (ITensor parameter : parameters) {
            parameterToPathMap.put(parameter, new Stack<>());
        }

        while (!toVisit.isEmpty()) {
            IGraph.IGraphNode node = toVisit.poll();
            boolean breathFirstSearchVisitedForAnyParameter = false;
            for (ITensor parameter : parametersToVisit) {
                parameterToPathMap.get(parameter).push(node);
                Set<IGraph.IGraphNode> breathFirstSearchVisited = visitedWhenSearchingForNode.get(parameter);
                if (!breathFirstSearchVisitedForAnyParameter) {
                    if (breathFirstSearchVisited.contains(node)) {
                        breathFirstSearchVisitedForAnyParameter = true;
                    }
                }
            }
            if (!breathFirstSearchVisitedForAnyParameter) {
                // We have reached a node that was not visited by the breath first search.
                // Thus, we can skip this node.
                continue;
            }
            if (node instanceof Graph.OperationGraphNode operationNode) {
                List<IGraph.IGraphNode> inputs = operationNode.getInputs();
                for (ITensor tensor : parametersToVisit) {
                    if (operationNode.getValue().isSame(tensor)) {
                        parametersToVisit.remove(tensor);
                        break;
                    }
                }
                if (parametersToVisit.isEmpty()) {
                    break;
                }
                // insert inputs at the beginning the nodesToVisit queue.
                // This ensures that we visit depth first
                for (int i = inputs.size() - 1; i >= 0; i--) {
                    toVisit.addFirst(inputs.get(i));
                }
            } else if (node instanceof Graph.TensorDeclarationGraphNode tensorDeclarationGraphNode) {
                for (ITensor tensor : parametersToVisit) {
                    if (tensorDeclarationGraphNode.getValue().isSame(tensor)) {
                        parametersToVisit.remove(tensor);
                        break;
                    } else {
                        Stack<IGraph.IGraphNode> pathToParameterStack = parameterToPathMap.get(tensor);
                        IGraph.IGraphNode removed = pathToParameterStack.pop();
                        if (pathToParameterStack.isEmpty()) {
                            continue;
                        }
                        IGraph.IGraphNode prev = pathToParameterStack.peek();
                        if (!(prev instanceof Graph.OperationGraphNode prevOp)) {
                            throw new IllegalStateException("Expected prev to be an operation node");
                        }
                        List<IGraph.IGraphNode> inputs = prevOp.getInputs();
                        if (inputs.indexOf(removed) == inputs.size() - 1) {
                            pathToParameterStack.pop(); // we just visited the last input of this operation, so we can pop it off the stack
                        }
                    }
                }
                if (parametersToVisit.isEmpty()) {
                    break;
                }
            }
        }
        return parameterToPathMap;
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


    @Override
    public void resetRecording() {
        valueToNodeMap.clear();
    }

}
