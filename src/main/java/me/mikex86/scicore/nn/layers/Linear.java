package me.mikex86.scicore.nn.layers;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.SciCore;
import me.mikex86.scicore.nn.IModule;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.ArrayList;
import java.util.List;

public class Linear implements IModule {

    @NotNull
    private final ISciCore sciCore;

    private final long inputSize;

    @NotNull
    private final ITensor weights;

    @Nullable
    private final ITensor bias;

    public Linear(@NotNull ISciCore sciCore, @NotNull DataType dataType, long inputSize, long outputSize, boolean useBias) {
        this.sciCore = sciCore;
        this.inputSize = inputSize;
        double k = 1.0 / Math.sqrt(inputSize);
        this.weights = sciCore.uniform(dataType, outputSize, inputSize).multiply(k).minus(-k);
        if (useBias) {
            this.bias = sciCore.uniform(dataType, outputSize).multiply(k).minus(-k);
        } else {
            this.bias = null;
        }
    }

    @Override
    public @NotNull ITensor forward(@NotNull ITensor input) {
        long[] inputShape = input.getShape();
        if (inputShape.length != 2) {
            throw new IllegalArgumentException("Input must be 2-dimensional for Linear layer");
        }
        if (inputShape[1] != inputSize) {
            throw new IllegalArgumentException("Input size must match the input size of the layer");
        }
        ITensor x = input.matmul(weights.transpose());
        if (bias != null) {
            x = x.plus(bias);
        }
        return x;
    }

    @Override
    public @NotNull List<ITensor> parameters() {
        List<ITensor> parameters = new ArrayList<>(2);
        parameters.add(weights);
        if (bias != null) {
            parameters.add(bias);
        }
        return parameters;
    }

}
