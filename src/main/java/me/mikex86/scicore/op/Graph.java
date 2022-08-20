package me.mikex86.scicore.op;

import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
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

    public Graph(@NotNull IGraphNode outputNode, @NotNull ISciCoreBackend backend) {
        this.outputNode = outputNode;
        this.backend = backend;
    }

    @Override
    @NotNull
    public IGraphNode getOutputNode() {
        return outputNode;
    }

    @Override
    public void requestGradientsFor(@NotNull ITensor... tensors) {
        for (ITensor tensor : tensors) {
            Optional<IGraphNode> nodeOpt = getNodeForTensor(tensor);
            if (nodeOpt.isEmpty()) {
                continue;
            }
            IGraphNode node = nodeOpt.get();

            if (!(node instanceof ITensorNodeWithGradient nodeWithGradient)) {
                throw new IllegalStateException("Requested gradients to be computed for node that can't hold a gradient: " + node);
            }
            nodeWithGradient.requestGradients();

            List<IGraphNode> downstreamNodes = getDependentNodes(node);

            for (IGraphNode downstreamNode : downstreamNodes) {
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
                    if (node.requiresGradients()) {
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
                if (tensorNode.getValue().equals(tensor)) {
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

    /**
     * Recursively lists downstream nodes of the specified node.
     *
     * @param node a given node
     * @return a list of all nodes that depend on the given node, meaning that they are a function of the given node. This list includes itself
     */
    @NotNull
    private List<IGraphNode> getDependentNodes(@NotNull IGraphNode node) {
        List<IGraphNode> nodes = new ArrayList<>();

        Queue<IGraphNode> nodesToVisit = new LinkedList<>();
        nodesToVisit.add(node);

        while (!nodesToVisit.isEmpty()) {
            IGraphNode currentNode = nodesToVisit.poll();
            nodes.add(currentNode);

            // traverse down the topology, if the graph extends downwards
            nodesToVisit.addAll(currentNode.getDownstreamNodes());
        }

        return nodes;
    }


    public static abstract class AbstractTensorNodeWithGradient extends AbstractGraphNode implements ITensorNodeWithGradient {

        private @Nullable ITensor gradient = null;

        private boolean requiresGradients = false;

        private boolean requestsGradients = false;

        @Override
        public void zeroGrad() {
            if (this.gradient != null) {
                this.gradient.fill(0);
            }
        }

        @Override
        public void deleteGradient() {
            this.gradient = null;
        }

        @Override
        public void accumulateGradient(@NotNull ITensor gradient) {
            if (this.gradient == null) {
                this.gradient = gradient;
            } else {
                Validator.assertTrue(ShapeUtils.equals(this.gradient.getShape(), gradient.getShape()), "Accumulative gradients must match shape");
                this.gradient = this.gradient.plus(gradient);
            }
        }

        @Nullable
        @Override
        public ITensor getGradient() {
            return gradient;
        }

        void setGradient(@Nullable ITensor gradient) {
            this.gradient = gradient;
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

    public static class ValueGraphNode extends AbstractGraphNode implements IValueNode {

        private final @NotNull Object value;

        public ValueGraphNode(@NotNull Object value) {
            this.value = value;
        }

        @NotNull
        public Object getValue() {
            return value;
        }
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

        private final @NotNull ITensor output;

        public OperationGraphNode(@NotNull OperationType operationType, @NotNull List<@NotNull IGraphNode> inputs, @NotNull ITensor output) {
            this.operationType = operationType;
            this.inputs = inputs;
            this.output = output;

            for (IGraphNode input : inputs) {
                input.addDownstreamNode(this); // indicate usage of input node by this node
            }
        }

        public @NotNull ITensor getOutput() {
            return output;
        }

        public @NotNull OperationType getOperationType() {
            return operationType;
        }

        public @NotNull List<@NotNull IGraphNode> getInputs() {
            return inputs;
        }

        @NotNull
        @Override
        public ITensor getValue() {
            return output;
        }

        @Override
        public void computeGradients() {
            ISciCoreBackend sciCoreBackend = getValue().getSciCoreBackend();
            sciCoreBackend.computeGradients(this);
        }

        @Override
        public String getName() {
            return operationType.name();
        }
    }
}
