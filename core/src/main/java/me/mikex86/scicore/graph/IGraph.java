package me.mikex86.scicore.graph;

import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.utils.dispose.IDisposable;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.*;

public interface IGraph extends IDisposable, AutoCloseable {

    @NotNull IGraphNode getOutputNode();

    /**
     * Requests gradients to be computed for the specified tensors
     *
     * @param tensors the list of tensors to compute gradients for
     */
    void requestGradientsFor(@NotNull List<ITensor> tensors);

    void backward();

    @NotNull Optional<ITensor> getGradient(@NotNull ITensor value);

    @Override
    default void close() {
        dispose();
    }

    interface IGraphNode {

        default String getName() {
            Random random = new Random(System.identityHashCode(this));
            char c1 = (char) ('A' + random.nextInt(26));
            char c2 = (char) ('A' + random.nextInt(26));
            return String.format("%c%c", c1, c2);
        }

        void addDownstreamNode(@NotNull IGraphNode node);

        @NotNull Set<IGraphNode> getDownstreamNodes();

    }

    abstract class AbstractGraphNode implements IGraphNode {

        /**
         * Set of graph nodes that use this node as an input
         */
        private final Set<IGraphNode> downstreamNodes = new HashSet<>();

        @Override
        public void addDownstreamNode(@NotNull IGraphNode node) {
            downstreamNodes.add(node);
        }

        @NotNull
        @Override
        public Set<IGraphNode> getDownstreamNodes() {
            return downstreamNodes;
        }
    }

    interface ITensorNode extends IGraphNode {
        @NotNull ITensor getValue();

        default String getName() {
            Random random = new Random(System.identityHashCode(getValue()));
            char c1 = (char) ('A' + random.nextInt(26));
            char c2 = (char) ('A' + random.nextInt(26));
            return String.format("%c%c", c1, c2);
        }

        void deleteValue();

        boolean hasValue();
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

    interface IDifferentiableNode extends ITensorNodeWithGradient {

        void computeGradients();

    }

}
