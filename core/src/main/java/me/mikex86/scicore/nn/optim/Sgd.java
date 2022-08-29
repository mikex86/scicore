package me.mikex86.scicore.nn.optim;

import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.op.IGraph;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public class Sgd implements IOptimizer {

    @NotNull
    private final ISciCore sciCore;

    private final float learningRate;
    @NotNull
    private final List<ITensor> parameters;

    private final boolean adaptiveLearningRate;

    private long nSteps = 1;

    private final float learningRateDecayFactor;

    public Sgd(@NotNull ISciCore sciCore, float learningRate, @NotNull List<ITensor> parameter) {
        this(sciCore, learningRate, parameter, true, 0.0f);
    }

    public Sgd(@NotNull ISciCore sciCore, float learningRate, @NotNull List<ITensor> parameters, boolean adaptiveLearningRate, float learningRateDecayFactor) {
        this.sciCore = sciCore;
        this.learningRate = learningRate;
        this.parameters = parameters;
        this.adaptiveLearningRate = adaptiveLearningRate;
        this.learningRateDecayFactor = learningRateDecayFactor;
    }

    @Override
    public void step(@NotNull IGraph graph) {
        graph.requestGradientsFor(parameters.toArray(new ITensor[0]));
        graph.backward();

        for (ITensor parameter : parameters) {
            ITensor gradient = graph.getGradient(parameter).orElseThrow(() -> new IllegalStateException("No gradient for parameter " + parameter));
            float learningRate = this.learningRate;
            if (adaptiveLearningRate) {
                learningRate *= (Math.pow(learningRateDecayFactor, nSteps));
            }
            ITensor newParameter = parameter.minus(gradient.multiply(learningRate)); // TODO: in-place operation
            parameter.setContents(newParameter);
        }
        nSteps++;
    }
}