package me.mikex86.scicore.op;

import me.mikex86.scicore.ITensor;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

public interface IGraph {
    @NotNull IGraphNode getOutputNode();

    void backward();

    interface IGraphNode {
    }

    interface IDifferentiableNode extends IGraphNode {

        void zeroGrad();

        void computeGradient();

        void accumulateGradient(@NotNull ITensor gradient);

        @Nullable ITensor getGradient();

        @NotNull ITensor getValue();

    }

}
