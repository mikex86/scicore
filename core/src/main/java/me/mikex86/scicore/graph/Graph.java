package me.mikex86.scicore.graph;

import me.mikex86.scicore.profiling.Profiler;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.backend.OperationRegistry;
import me.mikex86.scicore.graph.op.IDifferentiableOperation;
import me.mikex86.scicore.graph.op.IOperation;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.utils.OptionalUtils;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import me.mikex86.scicore.utils.dispose.IDisposable;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.*;
import java.util.function.Predicate;
import java.util.function.Supplier;

public class Graph implements IGraph {

    @NotNull
    private final IGraphNode outputNode;

    @NotNull
    private final ISciCoreBackend backend;

    @NotNull
    private final OperationRegistry operationRegistry;

    public Graph(@NotNull IGraphNode outputNode, @NotNull ISciCoreBackend backend, @NotNull OperationRegistry operationRegistry) {
        this.outputNode = outputNode;
        this.backend = backend;
        this.operationRegistry = operationRegistry;
    }

    @Override
    @NotNull
    public IGraphNode getOutputNode() {
        return outputNode;
    }

    @Override
    public void requestGradientsFor(@NotNull List<ITensor> parameters) {
        List<ITensorNodeWithGradient> parameterNodes = new ArrayList<>(parameters.size());
        for (ITensor parameter : parameters) {
            Optional<IGraphNode> nodeOpt = getNodeForTensor(parameter);
            if (nodeOpt.isEmpty()) {
                throw new IllegalArgumentException("Parameter not found in graph");
            }
            IGraphNode parameterNode = nodeOpt.get();
            if (!(parameterNode instanceof ITensorNodeWithGradient parameterNodeWithGradient)) {
                throw new IllegalArgumentException("Parameter is not a differentiable tensor");
            }
            parameterNodes.add(parameterNodeWithGradient);
        }
        for (ITensorNodeWithGradient parameterNode : parameterNodes) {
            parameterNode.requestGradients();
            parameterNode.setRequireGradients();
            Set<IGraphNode> downstreamNodes = parameterNode.getDownstreamNodes();
            Set<IGraphNode> visitedNodes = new HashSet<>();
            Queue<IGraphNode> queue = new ArrayDeque<>(downstreamNodes);
            while (!queue.isEmpty()) {
                IGraphNode node = queue.poll();
                if (visitedNodes.contains(node)) {
                    continue;
                }
                visitedNodes.add(node);
                if (node instanceof ITensorNodeWithGradient tensorNodeWithGradient) {
                    tensorNodeWithGradient.setRequireGradients();
                }
                queue.addAll(node.getDownstreamNodes());
            }
        }
    }

    @Override
    public void backward() {
        // initialize gradient to 1 because derivative of x in respect to itself is one. Duh.
        if (outputNode instanceof ITensorNodeWithGradient nodeWithGradient) {
            ITensor tensor = nodeWithGradient.getValue();
            if (!tensor.isScalar()) {
                throw new IllegalStateException("Cannot compute gradient of non-scalar tensor");
            }

            ITensor gradient = backend.createTensor(tensor.getDataType(), tensor.getShape());
            gradient.fill(1);
            nodeWithGradient.accumulateGradient(gradient); // dL/dL = 1

            // apply chain rule
            backPropagate(nodeWithGradient);

            // collect results
            Set<ITensor> gradients = collectGradientResults();

            // clear gradients
            clearUnusedGradients(gradients);
        } else {
            throw new IllegalStateException("Output node of graph must be differentiable!");
        }
    }

    /**
     * This function invokes .result() on all nodes that explicitly request gradients.
     * This is done such that we can delete the required, but not requested gradients from the graph, that would otherwise
     * have to be kept in memory until the lazy value of the requested gradient is computed.
     *
     * @return the set of computed gradients that were explicitly requested. Required but not requested gradients are not included in this set.
     */
    @NotNull
    private Set<ITensor> collectGradientResults() {
        Set<ITensor> gradients = Collections.newSetFromMap(new IdentityHashMap<>());

        Queue<IGraphNode> nodesToVisit = new LinkedList<>();
        nodesToVisit.add(outputNode);

        Set<IGraphNode> visited = new HashSet<>();

        while (!nodesToVisit.isEmpty()) {
            IGraphNode node = nodesToVisit.poll();
            if (visited.contains(node)) {
                continue;
            }
            visited.add(node);
            if (node instanceof ITensorNodeWithGradient nodeWithGradient) {
                if (nodeWithGradient.requestsGradients()) {
                    ITensor value = nodeWithGradient.getGradient();
                    if (value instanceof LazyTensor lazyTensor) {
                        lazyTensor.result(); // force computation of gradient
                        gradients.add(lazyTensor); // the instance in the set returned is the lazy wrapper
                    }
                }
                if (nodeWithGradient.requiresGradients()) {
                    if (node instanceof OperationGraphNode operationNode) {
                        nodesToVisit.addAll(operationNode.getInputs());
                    }
                }
            }
        }
        return gradients;
    }

    /**
     * Applies chain rule to all leaf nodes
     * This order of iteration guarantees topological ordering from root to leaf nodes
     * This is important because we need to compute the gradient the output node first before we can compute the
     * gradient of its inputs. This is required because we are applying the chain rule
     * which states dy/dx = dy/du * du/dx where u is a function of x.
     * An interpretation of this rule in the context of this graph is
     * that the gradient of an input node with respect to the output root node is the product of the gradient up
     * to the currently processed node and the gradient of the input node with respect to the currently processed node.
     */
    private void backPropagate(@NotNull ITensorNodeWithGradient node) {
        Deque<IGraphNode> topology = new LinkedList<>();
        Set<IGraphNode> visited = new HashSet<>();
        // build topology
        {
            buildTopo(node, topology, visited);
        }
        // back propagate
        for (IGraphNode currentNode : topology) {
            if (currentNode instanceof ITensorNodeWithGradient currentNodeWithGradient) {
                // only compute gradient for nodes for which it is required
                if (currentNodeWithGradient.requiresGradients()) {
                    if (currentNode instanceof IDifferentiableNode differentiableNode) {
                        differentiableNode.computeGradients();
                    }
                }
            }
        }
    }

    private void buildTopo(IGraphNode node, Deque<IGraphNode> topology, Set<IGraphNode> visited) {
        if (visited.contains(node)) {
            return;
        }
        visited.add(node);
        // This ordering guarantees that we don't use premature upstream gradients to compute subsequent gradients
        if (node instanceof OperationGraphNode operationNode) {
            for (IGraphNode input : operationNode.getInputs()) {
                buildTopo(input, topology, visited);
            }
            topology.addFirst(node); // add node AFTER all its inputs have been added
        }
    }

    /**
     * Clears the gradients of all nodes in the graph that don't require gradients
     */
    private void clearUnusedGradients(@NotNull Set<ITensor> gradients) {
        Queue<IGraphNode> nodesToVisit = new LinkedList<>();
        nodesToVisit.add(outputNode);

        Set<IGraphNode> visited = new HashSet<>();

        while (!nodesToVisit.isEmpty()) {
            IGraphNode node = nodesToVisit.poll();
            if (visited.contains(node)) {
                continue;
            }
            if (node instanceof ITensorNodeWithGradient nodeWithGradient) {
                if (!nodeWithGradient.requestsGradients()
                    && !gradients.contains(nodeWithGradient.getGradient())) { // we can't delete the gradient, if another node shares the same tensor instance as a gradient, eg. due to a plus operation
                    nodeWithGradient.deleteGradient();
                }
                if (node instanceof OperationGraphNode operationNode) {
                    // traverse up the topology, if the graph extends upwards
                    List<IGraphNode> inputs = operationNode.getInputs();
                    for (IGraphNode input : inputs) {
                        if (input instanceof ITensorNodeWithGradient inputWithGradient) {
                            if (inputWithGradient.requiresGradients()) {
                                nodesToVisit.add(input);
                            }
                        }
                    }
                }
            }
            visited.add(node);
        }
    }

    @Override
    public void dispose() {
        Queue<IGraphNode> nodesToVisit = new LinkedList<>();
        nodesToVisit.add(outputNode);

        Set<IGraphNode> visited = new HashSet<>();

        while (!nodesToVisit.isEmpty()) {
            IGraphNode node = nodesToVisit.poll();
            if (visited.contains(node)) {
                continue;
            }
            visited.add(node);
            if (node instanceof ITensorNodeWithGradient nodeWithGradient) {
                nodeWithGradient.deleteGradient();
            }
            if (node instanceof OperationGraphNode operationNode) {
                // traverse up the topology, if the graph extends upwards
                List<IGraphNode> inputs = operationNode.getInputs();
                nodesToVisit.addAll(inputs);
            }
        }
    }

    @NotNull
    public Optional<ITensor> getGradient(@NotNull ITensor tensor) {
        Optional<IGraphNode> node = getNodeForTensor(tensor);
        return OptionalUtils
                .cast(node, ITensorNodeWithGradient.class)
                .map(ITensorNodeWithGradient::getGradient)
                .filter(Predicate.not(ITensor::isDisposed));
    }

    @NotNull
    private Optional<IGraphNode> getNodeForTensor(@NotNull ITensor tensor) {
        Queue<IGraphNode> nodesToVisit = new LinkedList<>();
        nodesToVisit.add(outputNode);

        while (!nodesToVisit.isEmpty()) {
            IGraphNode node = nodesToVisit.poll();
            if (node instanceof ITensorNode tensorNode) {
                if (tensorNode.getValue() == tensor) {
                    return Optional.of(node);
                }
            }

            // traverse up the topology, if the graph extends upwards
            if (node instanceof OperationGraphNode operationNode) {
                nodesToVisit.addAll(operationNode.getInputs());
            }
        }

        return Optional.empty();
    }


    public static abstract class AbstractTensorNodeWithGradient extends AbstractGraphNode implements ITensorNodeWithGradient {

        private @Nullable ITensor upstreamGradient = null;


        private boolean requiresGradients = false;

        private boolean requestsGradients = false;

        @Override
        public void zeroGrad() {
            if (this.upstreamGradient != null) {
                this.upstreamGradient.fill(0);
            }
        }

        @Override
        public @Nullable ITensor getUpstreamGradient() {
            return this.upstreamGradient;
        }


        @Override
        public void deleteGradient() {
            if (this.upstreamGradient == null) {
                return;
            }
            if (this.upstreamGradient.isDisposed()) {
                return;
            }
            this.upstreamGradient.dispose();
            // don't set to null, because we want to know that this node has been disposed
            // otherwise we can't differentiate from zeroGrad
        }

        /**
         * Accumulates the tensor to the gradient of this node.
         *
         * @param gradient the gradient to accumulate. Do not auto-close this tensor!
         */
        @Override
        public void accumulateGradient(@NotNull ITensor gradient) {
            if (this.upstreamGradient == null) {
                this.upstreamGradient = gradient;
            } else {
                Validator.assertTrue(ShapeUtils.equals(this.upstreamGradient.getShape(), gradient.getShape()), "Accumulative gradients must match shape");
                this.upstreamGradient.add(gradient);
                gradient.close();
            }
        }

        @Nullable
        @Override
        public ITensor getGradient() {
            return upstreamGradient;
        }

        void setGradient(@Nullable ITensor gradient) {
            this.upstreamGradient = gradient;
        }

        @Override
        public void requestGradients() {
            this.requestsGradients = true;
        }

        @Override
        public boolean requestsGradients() {
            return requestsGradients;
        }

        @Override
        public void setRequireGradients() {
            this.requiresGradients = true;
        }

        @Override
        public boolean requiresGradients() {
            return requiresGradients;
        }
    }

    public static abstract class AbstractDifferentiableNode extends AbstractTensorNodeWithGradient implements IDifferentiableNode {

        @Override
        public abstract void computeGradients();

    }

    public static class TensorDeclarationGraphNode extends AbstractTensorNodeWithGradient {

        @Nullable
        private ITensor tensor;

        public TensorDeclarationGraphNode(@NotNull ITensor tensor) {
            this.tensor = tensor;
        }

        @NotNull
        @Override
        public ITensor getValue() {
            return Objects.requireNonNull(tensor, "Value has already been nullified");
        }

        @Override
        public void deleteValue() {
            this.tensor = null;
        }

        @Override
        public boolean hasValue() {
            return tensor != null;
        }

    }

    public static class OperationGraphNode extends AbstractDifferentiableNode {

        private final @NotNull OperationType operationType;

        private final @NotNull List<@NotNull IGraphNode> inputs;

        @Nullable
        private ITensor output;

        @NotNull
        private final IOperationContext operationContext;

        @NotNull
        private final OperationRegistry operationRegistry;

        private boolean gradEnabled;

        public OperationGraphNode(@NotNull OperationType operationType, @NotNull List<@NotNull IGraphNode> inputs, @NotNull IOperationContext operationContext, @NotNull OperationRegistry operationRegistry) {
            for (IGraphNode input : inputs) {
                input.addDownstreamNode(this);
            }
            this.operationType = operationType;
            this.inputs = inputs;
            this.operationContext = operationContext;
            this.operationRegistry = operationRegistry;
        }

        public void setOutput(@Nullable ITensor output) {
            this.output = output;
        }

        public @NotNull ITensor getOutput() {
            return Objects.requireNonNull(output, "Output not yet computed");
        }

        public @NotNull OperationType getOperationType() {
            return operationType;
        }

        public @NotNull List<@NotNull IGraphNode> getInputs() {
            return inputs;
        }

        @NotNull
        public IOperationContext getOperationContext() {
            return operationContext;
        }

        @NotNull
        @Override
        public ITensor getValue() {
            return getOutput();
        }

        @Override
        public void computeGradients() {
            OperationType operationType = getOperationType();
            IOperation operation = operationRegistry.getOperation(operationType);
            if (!(operation instanceof IDifferentiableOperation differentiableOperation)) {
                throw new IllegalStateException("Operation is not differentiable: " + operationType);
            }
            differentiableOperation.computeGradients(this);
        }

        @Override
        public String getName() {
            return operationType.name();
        }

        @Override
        public void deleteValue() {
            this.output = null;
        }

        @Override
        public boolean hasValue() {
            return output != null;
        }

        /**
         * Performs the operation of this node.
         */
        @NotNull
        public ITensor perform() {
            OperationType operationType = getOperationType();
            IOperation operation = operationRegistry.getOperation(operationType);
            List<ITensor> inputTensors = new ArrayList<>(inputs.size());
            for (IGraphNode input : inputs) {
                if (input instanceof ITensorNode tensorNode) {
                    inputTensors.add(tensorNode.getValue());
                } else {
                    throw new IllegalStateException("Input node is not a tensor node: " + input);
                }
            }
            return operation.perform(this.operationContext, inputTensors);
        }

        public boolean hasOutput() {
            return output != null;
        }

        public void replaceInputs(@NotNull IGraphNode original, @NotNull IGraphNode replacement) {
            for (int i = 0; i < inputs.size(); i++) {
                if (inputs.get(i) == original) {
                    inputs.set(i, replacement);
                }
            }
        }

        public void setEnableGrad(boolean gradEnabled) {
            this.gradEnabled = gradEnabled;
        }

        public boolean isGradEnabled() {
            return gradEnabled;
        }
    }

    public interface IOperationContext {

        void saveForBackward(@NotNull String name, @NotNull ITensor tensor);

        @NotNull
        Optional<ITensor> getSavedTensor(@NotNull String name);

        @NotNull OptionBundle getOptionBundle();

        @NotNull ITensor getSavedTensorOrPopulateWith(@NotNull String name, @NotNull Supplier<ITensor> defaultSupplier);

        boolean hasSavedTensor(@NotNull String name);

        @NotNull Map<String, ITensor> getSavedTensors();
    }

    public static class OperationContext implements IOperationContext {

        @NotNull
        private final Map<String, ITensor> savedTensors = new HashMap<>();

        @NotNull
        private final OptionBundle optionBundle;

        public OperationContext(@NotNull OptionBundle optionBundle) {
            this.optionBundle = optionBundle;
        }

        @Override
        public void saveForBackward(@NotNull String name, @NotNull ITensor tensor) {
            savedTensors.put(name, tensor);
        }

        @Override
        public @NotNull Optional<ITensor> getSavedTensor(@NotNull String name) {
            return Optional.ofNullable(savedTensors.get(name));
        }

        @NotNull
        @Override
        public OptionBundle getOptionBundle() {
            return optionBundle;
        }

        @Override
        public @NotNull ITensor getSavedTensorOrPopulateWith(@NotNull String name, @NotNull Supplier<ITensor> defaultSupplier) {
            return savedTensors.computeIfAbsent(name, s -> defaultSupplier.get());
        }

        @Override
        public boolean hasSavedTensor(@NotNull String name) {
            return savedTensors.containsKey(name);
        }

        @Override
        public @NotNull Map<String, ITensor> getSavedTensors() {
            return savedTensors;
        }
    }
}
