package me.mikex86.scicore.graph;

import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.backend.OperationRegistry;
import me.mikex86.scicore.graph.op.IDifferentiableOperation;
import me.mikex86.scicore.graph.op.IOperation;
import me.mikex86.scicore.utils.OptionalUtils;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.*;

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
        for (ITensor parameter : parameters) {
            Optional<IGraphNode> nodeOpt = Optional.empty();
            Stack<IGraphNode> pathToNode = new Stack<>();
            {
                Deque<IGraphNode> nodesToVisit = new LinkedList<>();
                nodesToVisit.add(outputNode);

                while (!nodesToVisit.isEmpty()) {
                    IGraphNode node = nodesToVisit.poll();
                    pathToNode.push(node);
                    if (node instanceof OperationGraphNode operationNode) {
                        if (operationNode.getValue().isSame(parameter)) {
                            nodeOpt = Optional.of(node);
                            break;
                        }
                        // traverse up the topology, if the graph extends upwards
                        List<IGraphNode> inputs = operationNode.getInputs();

                        // insert inputs at the beginning the nodesToVisit queue
                        // This ensures that we visit depth first
                        for (int i = inputs.size() - 1; i >= 0; i--) {
                            nodesToVisit.addFirst(inputs.get(i));
                        }
                    } else if (node instanceof TensorDeclarationGraphNode tensorDeclarationNode) {
                        if (tensorDeclarationNode.getValue().isSame(parameter)) {
                            nodeOpt = Optional.of(node);
                            break;
                        } else {
                            IGraphNode removed = pathToNode.pop();
                            IGraphNode prev = pathToNode.peek();
                            if (!(prev instanceof OperationGraphNode prevOp)) {
                                throw new IllegalStateException("Unexpected node type: " + prev.getClass().getName());
                            }
                            List<IGraphNode> prevOpInputs = prevOp.getInputs();
                            int index = prevOpInputs.indexOf(removed);
                            if (index == prevOpInputs.size() - 1) {
                                pathToNode.pop(); // we just visited the last input of this operation, so we can pop it off the stack
                            }
                        }
                    }
                }
            }
            if (nodeOpt.isEmpty()) {
                throw new IllegalArgumentException("Tensor is not part of the graph");
            }
            IGraphNode node = nodeOpt.get();

            ITensorNodeWithGradient nodeWithGradient = (ITensorNodeWithGradient) node;
            nodeWithGradient.requestGradients();

            for (IGraphNode downstreamNode : pathToNode) {
                if (!(downstreamNode instanceof ITensorNodeWithGradient downStreamNodeWithGradient)) {
                    throw new IllegalArgumentException("Requested gradient for tensor that cannot hold a gradient: " + downstreamNode);
                }
                downStreamNodeWithGradient.setRequireGradients(); // all nodes that depend on this node will have their gradients computed for them
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
            ITensor gradient;
            {
                // ones like tensor
                gradient = backend.createTensor(tensor.getDataType(), tensor.getShape());
                gradient.fill(1);
            }
            nodeWithGradient.accumulateGradient(gradient);

            // apply chain rule
            backPropagate(nodeWithGradient);
        } else {
            throw new IllegalStateException("Output node of graph must be differentiable!");
        }
    }


    /**
     * Recursively applies chain rule to all leaf nodes
     * This order of iteration guarantees topological ordering from root to leaf nodes
     * This is important because we need to compute the gradient the output node first before we can compute the
     * gradient of its inputs. This is required because we are applying the chain rule
     * which states dy/dx = dy/du * du/dx where u is a function of x.
     * An interpretation of this rule in the context of this graph is
     * that the gradient of an input node with respect to the output root node is the product of the gradient up
     * to the currently processed node and the gradient of the input node with respect to the currently processed node.
     */
    private void backPropagate(@NotNull ITensorNodeWithGradient node) {
        // only compute gradient for nodes for which it is required
        if (node instanceof IDifferentiableNode differentiableNode) {
            differentiableNode.computeGradients();
        }
        // traverse up the topology, if the graph extends upwards
        if (node instanceof OperationGraphNode operationNode) {
            for (IGraphNode inputNode : operationNode.getInputs()) {
                if (inputNode instanceof ITensorNodeWithGradient inputNodeWithGradient) {
                    if (inputNodeWithGradient.requiresGradients()) {
                        backPropagate(inputNodeWithGradient);
                    }
                }
            }
        }
        if (!node.requestsGradients()) {
            node.deleteGradient();
        }
    }

    @NotNull
    public Optional<ITensor> getGradient(@NotNull ITensor tensor) {
        Optional<IGraphNode> node = getNodeForTensor(tensor);
        return OptionalUtils.cast(node, ITensorNodeWithGradient.class).map(ITensorNodeWithGradient::getGradient);
    }

    @NotNull
    private Optional<IGraphNode> getNodeForTensor(@NotNull ITensor tensor) {
        Queue<IGraphNode> nodesToVisit = new LinkedList<>();
        nodesToVisit.add(outputNode);

        while (!nodesToVisit.isEmpty()) {
            IGraphNode node = nodesToVisit.poll();
            if (node instanceof ITensorNode tensorNode) {
                if (tensorNode.getValue().isSame(tensor)) {
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
            this.upstreamGradient = null;
        }

        @Override
        public void accumulateGradient(@NotNull ITensor gradient) {
            if (this.upstreamGradient == null) {
                this.upstreamGradient = gradient;
            } else {
                Validator.assertTrue(ShapeUtils.equals(this.upstreamGradient.getShape(), gradient.getShape()), "Accumulative gradients must match shape");
                this.upstreamGradient.add(gradient);
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

        private final @NotNull ITensor tensor;

        public TensorDeclarationGraphNode(@NotNull ITensor tensor) {
            this.tensor = tensor;
        }

        @NotNull
        @Override
        public ITensor getValue() {
            return tensor;
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

        public OperationGraphNode(@NotNull OperationType operationType, @NotNull List<@NotNull IGraphNode> inputs, @NotNull IOperationContext operationContext, @NotNull OperationRegistry operationRegistry) {
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


        /**
         * Performs the operation of this node.
         */
        @NotNull
        public ITensor perform() {
            OperationType operationType = getOperationType();
            IOperation operation = operationRegistry.getOperation(operationType);
            List<ITensor> inputTensors = new ArrayList<>();
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
    }

    public interface IOperationContext {

        void saveForBackward(@NotNull String name, @NotNull ITensor tensor);

        @NotNull
        Optional<ITensor> getSavedTensor(@NotNull String name);

        @NotNull OptionBundle getOptionBundle();
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
    }
}
