package me.mikex86.scicore.op;

import me.mikex86.scicore.ITensor;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.*;

public interface IGraph {

    @NotNull IGraphNode getOutputNode();

    /**
     * Requests gradients to be computed for the specified tensors
     *
     * @param tensors the list of tensors to compute gradients for
     */
    void requestGradientsFor(@NotNull ITensor... tensors);

    void backward();

    @NotNull Optional<ITensor> getGradient(@NotNull ITensor value);

    /**
     * Recursively lists downstream nodes of the specified node.
     *
     * @param node a given node
     * @return a list of all nodes that depend on the given node, meaning that they are a function of the given node. This list includes itself
     */
    @NotNull List<IGraphNode> getDependentNodes(@NotNull IGraphNode node);

    interface IGraphNode {

        default String getName() {
            Random random = new Random(hashCode());
            char randomChar = (char) (random.nextInt(26) + 'a');
            return String.valueOf(randomChar);
        }

        /**
         * Adds a downstream node to this node. This indicates a usage of this node by another node, which takes it as an input.
         *
         * @param downstreamNode the downstream node
         */
        void addDownstreamNode(@NotNull IGraphNode downstreamNode);

        /**
         * @return the list of all usages of this node. Empty for the output node.
         */
        @NotNull
        List<IGraphNode> getDownstreamNodes();

        @NotNull
        IGraphNode deepCopy();

    }

    abstract class AbstractGraphNode implements IGraphNode {

        @NotNull
        private final List<IGraphNode> downstreamNodes = new ArrayList<>();

        @Override
        public void addDownstreamNode(@NotNull IGraphNode downstreamNode) {
            downstreamNodes.add(downstreamNode);
        }

        @Override
        public @NotNull List<IGraphNode> getDownstreamNodes() {
            return downstreamNodes;
        }
    }

    interface ITensorNode extends IGraphNode {
        @NotNull ITensor getValue();
    }

    interface ITensorNodeWithGradient extends ITensorNode {

        /**
         * Sets the gradient for this tensor to zero.
         */
        void zeroGrad();

        void accumulateGradient(@NotNull ITensor gradient);

        @Nullable ITensor getGradient();

        @Nullable ITensor getUpstreamGradient();

        /**
         * Requests gradients to be computed for this node. This marks an explicit request for the autograd engine
         * to compute gradients for this node, as well as save the gradient for this node.
         */
        void requestGradients();

        /**
         * @return true if gradients have been requested for this node. This does not mean gradients don't have to be computed though, as gradients up to this node might
         * still be required as downstream gradients.
         */
        boolean requestsGradients();

        /**
         * Marks this node as needing gradients to be computed for. This can either mean it has been explicitly requested, or it is a node that is a function of
         * another node that gradients have been explicitly requested for.
         * If the gradient is only required as a downstream gradient, the gradient will be deleted after the backward pass.
         */
        void setRequireGradients();

        /**
         * @return true if this node requires gradients to be computed for.
         * @see #setRequireGradients()
         */
        boolean requiresGradients();

        /**
         * Deletes the gradient tensor, which results in resources being freed, if no other references exist to it.
         * This is different to zeroGrad(), which just sets the gradient to zero.
         */
        void deleteGradient();
    }

    interface IValueNode extends IGraphNode {
        @NotNull Object getValue();
    }

    interface IDifferentiableNode extends ITensorNodeWithGradient {

        void computeGradients();

    }

}
