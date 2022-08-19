package me.mikex86.scicore.op;

import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
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
    private final Map<ITensor, IDifferentiableNode> gradientMap = new IdentityHashMap<>();

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
    public void backward() {
        this.gradientMap.clear();

        // TODO: Implement "requires_grad" boolean and "should_save_grad" boolean

        Queue<IGraphNode> nodes = new LinkedList<>();
        Set<IGraphNode> visited = Collections.newSetFromMap(new IdentityHashMap<>()); // compare identity, not contents

        // Handle output node
        // Defines root note that we want to derive with respect to
        {
            // initialize gradient to 1 because derivative of x in respect to itself is one. Duh.
            if (outputNode instanceof AbstractDifferentiableNode differentiableNode) {
                ITensor tensor = differentiableNode.getValue();
                if (!tensor.isScalar()) {
                    throw new IllegalStateException("Cannot compute gradient of non-scalar tensor");
                }
                ITensor gradient;
                {
                    // ones like tensor
                    gradient = backend.createTensor(tensor.getDataType(), tensor.getShape());
                    gradient.fill(1);
                }
                differentiableNode.accumulateGradient(gradient);
                nodes.add(differentiableNode);
            } else {
                throw new IllegalStateException("Output node of graph must be differentiable!");
            }
        }

        // Recursively apply chain rule to all leaf nodes
        // This order of iteration guarantees topological ordering from root to leaf nodes
        // This is important because we need to compute the gradient the output node first before we can compute the
        // gradient of its inputs. This is required because we are applying the chain rule
        // which states dy/dx = dy/du * du/dx where u is a function of x.
        // An interpretation of this rule in the context of this graph is
        // that the gradient of an input node with respect to the output root node is the product of the gradient up
        // to the currently processed node and the gradient of the input node with respect to the currently processed node.
        while (!nodes.isEmpty()) {
            IGraphNode node = nodes.poll();

            if (node instanceof IDifferentiableNode differentiableNode) {
                differentiableNode.computeGradients();
                this.gradientMap.put(differentiableNode.getValue(), differentiableNode);
            }

            // traverse up the topology, if the graph extends upwards
            if (node instanceof OperationGraphNode operationNode) {
                for (IGraphNode inputNode : operationNode.getInputs()) {
                    if (!visited.contains(inputNode)) {
                        nodes.add(inputNode);
                    }
                }
            }
            visited.add(node);
        }
    }

    @NotNull
    public Optional<ITensor> getGradient(@NotNull ITensor tensor) {
        IDifferentiableNode node = this.gradientMap.get(tensor);
        return Optional.ofNullable(node).map(IDifferentiableNode::getGradient);
    }

    public static abstract class AbstractDifferentiableNode implements IDifferentiableNode {

        private @Nullable ITensor gradient = null;

        @Override
        public void zeroGrad() {
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
    }

    public static class ValueGraphNode implements IGraphNode {

        private final @NotNull Object value;

        public ValueGraphNode(@NotNull Object value) {
            this.value = value;
        }

        @NotNull
        public Object getValue() {
            return value;
        }
    }

    public static class TensorDeclarationGraphNode extends AbstractDifferentiableNode {

        private final @NotNull ITensor tensor;

        public TensorDeclarationGraphNode(@NotNull ITensor tensor) {
            this.tensor = tensor;
        }

        @NotNull
        @Override
        public ITensor getValue() {
            return tensor;
        }

        @Override
        public void computeGradients() {
            // TODO: REMOVE
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
