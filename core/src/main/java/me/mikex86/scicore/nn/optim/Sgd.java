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

    private final float initialLearningRate;


    @NotNull
    private final List<ITensor> parameters;

    private final boolean adaptiveLearningRate;

    private long nSteps = 0;

    private final float learningRateDecayFactor;

    public Sgd(@NotNull ISciCore sciCore, float learningRate, @NotNull List<ITensor> parameter) {
        this(sciCore, learningRate, false, 0.0f, parameter);
    }

    public Sgd(@NotNull ISciCore sciCore, float initialLearningRate, float endLearningRate, long nStepsUntilEndLearningRateReached, @NotNull List<ITensor> parameter) {
        this(
                sciCore, initialLearningRate,
                true,
                (float) Math.pow(endLearningRate / initialLearningRate, 1.0 / nStepsUntilEndLearningRateReached),
                parameter
        );
    }

    private Sgd(@NotNull ISciCore sciCore, float initialLearningRate, boolean adaptiveLearningRate, float learningRateDecayFactor, @NotNull List<ITensor> parameters) {
        this.sciCore = sciCore;
        this.initialLearningRate = initialLearningRate;
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
                        float learningRate;
                        if (adaptiveLearningRate) {
                            learningRate = (float) (initialLearningRate * Math.pow(learningRateDecayFactor, nSteps));
                        } else {
                            learningRate = this.initialLearningRate;
                        }
                        try (ITensor scaledGradient = gradient.multiply(learningRate)) {
                            try (ITensor newGradient = parameter.minus(scaledGradient)) {
                                parameter.setContents(newGradient);
                            }
                        }
                    }
                }
                return null;
            });
            nSteps++;
        }
    }
}
