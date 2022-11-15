package me.mikex86.scicore.nn.optim;

import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.tensor.LazyTensor;
import org.jetbrains.annotations.NotNull;

import java.util.List;
import java.util.concurrent.Callable;

public class Sgd implements IOptimizer {

    @NotNull
    private final ISciCore sciCore;

    private final float learningRate;
    @NotNull
    private final List<ITensor> parameters;

    private final boolean adaptiveLearningRate;

    private long nSteps = 0;

    private final float learningRateDecayFactor;

    public Sgd(@NotNull ISciCore sciCore, float learningRate, @NotNull List<ITensor> parameter) {
        this(sciCore, learningRate, parameter, false, 0.0f);
    }

    public Sgd(@NotNull ISciCore sciCore, float learningRate, @NotNull List<ITensor> parameters, boolean adaptiveLearningRate, float learningRateDecayFactor) {
        this.sciCore = sciCore;
        this.learningRate = learningRate;
        this.parameters = parameters;
        this.adaptiveLearningRate = adaptiveLearningRate;
        this.learningRateDecayFactor = learningRateDecayFactor;
    }

    @Override
    public void step(@NotNull ITensor loss) {
        try (IGraph graph = sciCore.getBackpropagationGraphUpTo(loss, parameters)) {
            sciCore.getBackend().getOperationRecorder().recordWithScope(() -> {
                graph.backward();
                for (ITensor parameter : parameters) {
                    try (ITensor gradient = graph.getGradient(parameter)
                            .orElseThrow(() -> new IllegalStateException("No gradient for parameter"))) {
                        float learningRate = this.learningRate;
                        if (adaptiveLearningRate) {
                            learningRate *= (Math.pow(1f - learningRateDecayFactor, nSteps));
                        }
                        try (ITensor scaledGradient = gradient.multiply(learningRate)) {
                            parameter.subtract(scaledGradient);
                        }
                    }
                }
                return null;
            });
            nSteps++;
        }
    }
}
