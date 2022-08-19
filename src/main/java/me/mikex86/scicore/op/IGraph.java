package me.mikex86.scicore.op;

import me.mikex86.scicore.ITensor;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.Optional;
import java.util.Random;

public interface IGraph {

    @NotNull IGraphNode getOutputNode();

    void backward();

    @NotNull Optional<ITensor> getGradient(@NotNull ITensor value);

    interface IGraphNode {

        default String getName() {
            Random random = new Random(hashCode());
            char randomChar = (char) (random.nextInt(26) + 'a');
            return String.valueOf(randomChar);
        }
    }

    interface IDifferentiableNode extends IGraphNode {

        void zeroGrad();

        void computeGradients();

        void accumulateGradient(@NotNull ITensor gradient);

        @Nullable ITensor getGradient();

        @NotNull ITensor getValue();

    }

}
