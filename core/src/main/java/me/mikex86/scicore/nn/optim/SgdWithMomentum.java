package me.mikex86.scicore.nn.optim;

import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.tensor.ITensor;
import org.jetbrains.annotations.NotNull;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

public class SgdWithMomentum implements IOptimizer {

    @NotNull
    private final ISciCore sciCore;

    private final float initialLearningRate;

    @NotNull
    private final List<ITensor> parameters;

    private final boolean adaptiveLearningRate;

    private long nSteps = 0;

    private final float learningRateDecayFactor;

    private final float momentumCoefficient;

    private final float dampeningFactor;

    public SgdWithMomentum(@NotNull ISciCore sciCore, float learningRate, float momentumCoefficient, float dampeningFactor, @NotNull List<ITensor> parameters) {
        this(sciCore, learningRate, false, 0.0f, momentumCoefficient, dampeningFactor, parameters);
    }

    public SgdWithMomentum(@NotNull ISciCore sciCore, float initialLearningRate, float endLearningRate, long nStepsUntilEndLearningRateReached, float momentumCoefficient, @NotNull List<ITensor> parameters, float dampeningFactor) {
        this(
                sciCore, initialLearningRate,
                true,
                (float) Math.pow(endLearningRate / initialLearningRate, 1.0 / nStepsUntilEndLearningRateReached),
                momentumCoefficient, dampeningFactor, parameters
        );
    }

    private SgdWithMomentum(@NotNull ISciCore sciCore, float initialLearningRate, boolean adaptiveLearningRate, float learningRateDecayFactor, float momentumCoefficient, float dampeningFactor, @NotNull List<ITensor> parameters) {
        this.sciCore = sciCore;
        this.initialLearningRate = initialLearningRate;
        this.dampeningFactor = dampeningFactor;
        this.parameters = parameters;
        this.adaptiveLearningRate = adaptiveLearningRate;
        this.learningRateDecayFactor = learningRateDecayFactor;
        this.momentumCoefficient = momentumCoefficient;
    }

    /**
     * Maps parameters to their last momentum tensors at t-1.
     */
    @NotNull
    private final Map<ITensor, ITensor> lastMomentumTensors = new IdentityHashMap<>();

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
                        ITensor lastMomentumTensor = lastMomentumTensors.get(parameter);
                        if (lastMomentumTensor == null) {
                            lastMomentumTensor = sciCore.zerosLike(parameter);
                            lastMomentumTensors.put(parameter, lastMomentumTensor);
                        }
                        try (ITensor scaledMomentum = lastMomentumTensor.multiply(momentumCoefficient);
                             ITensor momentumScaledGradient = gradient.multiply(1f - dampeningFactor);
                             ITensor momentumTensor = scaledMomentum.plus(momentumScaledGradient)) {
                            try (ITensor scaledMomentumTensor = momentumTensor.multiply(learningRate)) {
                                parameter.subtract(scaledMomentumTensor);
                            }
                            lastMomentumTensor.setContents(momentumTensor);
                        }
                    }
                }
                return null;
            });
            nSteps++;
        }
    }
}
