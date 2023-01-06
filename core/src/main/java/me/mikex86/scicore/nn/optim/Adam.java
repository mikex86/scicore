package me.mikex86.scicore.nn.optim;

import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.tensor.ITensor;
import org.jetbrains.annotations.NotNull;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

public class Adam implements IOptimizer {

    @NotNull
    private final ISciCore sciCore;

    private final boolean adaptiveLearningRate;

    private final float initialLearningRate;

    private final float learningRateDecayFactor;

    @NotNull
    private final List<ITensor> parameters;

    private int nSteps = 0;

    private final float beta1;

    private final float beta2;

    public Adam(@NotNull ISciCore sciCore, float learningRate,  float beta1, float beta2, @NotNull List<ITensor> parameters) {
        this(sciCore, learningRate, false, 0.0f, beta1, beta2, parameters);
    }

    public Adam(@NotNull ISciCore sciCore, float initialLearningRate, float endLearningRate, long nStepsUntilEndLearningRateReached, float beta1, float beta2, @NotNull List<ITensor> parameters) {
        this(
                sciCore, initialLearningRate,
                true,
                (float) Math.pow(endLearningRate / initialLearningRate, 1.0 / nStepsUntilEndLearningRateReached),
                beta1, beta2, parameters
        );
    }

    public Adam(@NotNull ISciCore sciCore, float initialLearningRate, boolean adaptiveLearningRate, float learningRateDecayFactor, float beta1, float beta2, @NotNull List<ITensor> parameters) {
        this.sciCore = sciCore;
        this.initialLearningRate = initialLearningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.parameters = parameters;
        this.adaptiveLearningRate = adaptiveLearningRate;
        this.learningRateDecayFactor = learningRateDecayFactor;
    }

    /**
     * Maps parameters to their first moment (momentum).
     */
    private final Map<ITensor, ITensor> lastFirstMoments = new IdentityHashMap<>();

    /**
     * Maps parameters to their second moment (RMSProp).
     */
    private final Map<ITensor, ITensor> lastSecondMoments = new IdentityHashMap<>();

    private static final float EPSILON = 1e-8f;

    @Override
    public void step(@NotNull ITensor loss) {
        try (IGraph graph = sciCore.getBackpropagationGraphUpTo(loss, parameters)) {
            sciCore.getBackend().getOperationRecorder().recordWithScope(() -> {
                graph.backward();
                for (ITensor parameter : parameters) {
                    try (ITensor gradients = graph.getGradient(parameter)
                            .orElseThrow(() -> new IllegalStateException("No gradient for parameter"))) {
                        float learningRate = getLearningRate();

                        ITensor firstMoment = lastFirstMoments.get(parameter);
                        if (firstMoment == null) {
                            firstMoment = sciCore.zerosLike(parameter);
                            lastFirstMoments.put(parameter, firstMoment);
                        }

                        ITensor secondMoment = lastSecondMoments.get(parameter);
                        if (secondMoment == null) {
                            secondMoment = sciCore.zerosLike(parameter);
                            lastSecondMoments.put(parameter, secondMoment);
                        }

                        try (ITensor scaledMomentum = firstMoment.multiply(beta1);
                             ITensor scaledGradients = gradients.multiply(1.0f - beta1);
                             ITensor momentumTensor = scaledMomentum.plus(scaledGradients)) {
                            firstMoment.setContents(momentumTensor);
                        }

                        try (ITensor scaledRMSProp = secondMoment.multiply(beta2);
                             ITensor scaledGradients = gradients.pow(2.0f).multiply(1.0f - beta2);
                             ITensor rmsPropTensor = scaledRMSProp.plus(scaledGradients)) {
                            secondMoment.setContents(rmsPropTensor);
                        }

                        try (ITensor biasCorrectedFirstMoment = firstMoment.divide(1.0f - (float) Math.pow(beta1, nSteps + 1));
                             ITensor biasCorrectedSecondMoment = secondMoment.divide(1.0f - (float) Math.pow(beta2, nSteps + 1));
                             ITensor biasCorrectedSecondMomentSqrt = biasCorrectedSecondMoment.pow(0.5f);
                             ITensor biasCorrectedSecondMomentSqrtPlusEpsilon = biasCorrectedSecondMomentSqrt.plus(EPSILON);
                             ITensor gradientsWithMoments = biasCorrectedFirstMoment.divide(biasCorrectedSecondMomentSqrtPlusEpsilon);
                             ITensor scaledGradientsWithMoments = gradientsWithMoments.multiply(learningRate)) {
                            parameter.subtract(scaledGradientsWithMoments);
                        }
                    }
                }
                return null;
            });
            nSteps++;
        }
    }

    private float getLearningRate() {
        return adaptiveLearningRate ? initialLearningRate * (float) Math.pow(learningRateDecayFactor, nSteps) : initialLearningRate;
    }
}
