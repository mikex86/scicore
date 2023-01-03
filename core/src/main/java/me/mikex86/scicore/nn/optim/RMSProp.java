package me.mikex86.scicore.nn.optim;

import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.tensor.ITensor;
import org.jetbrains.annotations.NotNull;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

public class RMSProp implements IOptimizer {

    @NotNull
    private final ISciCore sciCore;

    private final float initialLearningRate;

    @NotNull
    private final List<ITensor> parameters;

    private final boolean adaptiveLearningRate;

    private long nSteps = 0;

    private final float learningRateDecayFactor;

    private final float rmsDecayFactor;

    public RMSProp(@NotNull ISciCore sciCore, float learningRate, float rmsDecayFactor, @NotNull List<ITensor> parameters) {
        this(sciCore, learningRate, false, 0.0f, rmsDecayFactor, parameters);
    }

    public RMSProp(@NotNull ISciCore sciCore, float initialLearningRate, float endLearningRate, long nStepsUntilEndLearningRateReached, float rmsDecayFactor, @NotNull List<ITensor> parameters) {
        this(
                sciCore, initialLearningRate,
                true,
                (float) Math.pow(endLearningRate / initialLearningRate, 1.0 / nStepsUntilEndLearningRateReached),
                rmsDecayFactor, parameters
        );
    }

    private RMSProp(@NotNull ISciCore sciCore, float initialLearningRate, boolean adaptiveLearningRate, float learningRateDecayFactor, float rmsDecayFactor, @NotNull List<ITensor> parameters) {
        this.sciCore = sciCore;
        this.initialLearningRate = initialLearningRate;
        this.rmsDecayFactor = rmsDecayFactor;
        this.parameters = parameters;
        this.adaptiveLearningRate = adaptiveLearningRate;
        this.learningRateDecayFactor = learningRateDecayFactor;
    }

    private final Map<ITensor, ITensor> lastMovingAvgs = new IdentityHashMap<>();

    private final float epsilon = 1e-8f;

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
                        ITensor movingAvg = lastMovingAvgs.get(parameter);
                        if (movingAvg == null) {
                            movingAvg = sciCore.zerosLike(parameter);
                            lastMovingAvgs.put(parameter, movingAvg);
                        }
                        try (ITensor scaledOldMovingAvg = movingAvg.multiply(rmsDecayFactor);
                             ITensor squaredGradient = gradient.pow(2.0f);
                             ITensor scaledNewMovingAvg = squaredGradient.multiply(1.0f - rmsDecayFactor);
                             ITensor newMovingAvg = scaledOldMovingAvg.plus(scaledNewMovingAvg)) {
                            movingAvg.setContents(newMovingAvg);
                        }
                        try (ITensor movingAvgSqrt = movingAvg.pow(0.5f);
                             ITensor movingAvgSqrtPlusEps = movingAvgSqrt.plus(epsilon);
                             ITensor rmsScaledGradients = gradient.divide(movingAvgSqrtPlusEps);
                             ITensor lrScaledGradients = rmsScaledGradients.multiply(learningRate)) {
                            parameter.subtract(lrScaledGradients);
                        }
                    }
                }
                return null;
            });
            nSteps++;
        }
    }

}
