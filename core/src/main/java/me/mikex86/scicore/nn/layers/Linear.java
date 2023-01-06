package me.mikex86.scicore.nn.layers;

import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.nn.IModule;
import me.mikex86.scicore.tensor.LazyTensor;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Linear implements IModule {

    private final long inputSize;

    @NotNull
    private final ITensor weights;

    @Nullable
    private final ITensor bias;
    @NotNull
    private final List<ITensor> parameters;

    public Linear(@NotNull ISciCore sciCore, @NotNull DataType dataType, long inputSize, long outputSize, boolean useBias) {
        this.inputSize = inputSize;
        float k = (float) (1.0 / Math.sqrt(inputSize));
        this.weights = ((LazyTensor) sciCore.uniform(dataType, outputSize, inputSize).multiply(2f * k).minus(k)).result();
        if (useBias) {
            this.bias = ((LazyTensor) sciCore.uniform(dataType, outputSize).multiply(2f * k).minus(k)).result();
        } else {
            this.bias = null;
        }
        this.parameters = new ArrayList<>(2);
        this.parameters.add(this.weights);
        if (this.bias != null) {
            this.parameters.add(this.bias);
        }
    }

    @Override
    public @NotNull ITensor forward(@NotNull ITensor input) {
        long[] inputShape = input.getShape();
        if (inputShape.length != 2 && inputShape.length != 3) {
            throw new IllegalArgumentException("Input must be a 2D or 3D tensor");
        }
        if (inputShape[inputShape.length - 1] != inputSize) {
            throw new IllegalArgumentException("Input size must match the input size of the layer");
        }
        try (ITensor x = input.matmul(weights, false, true)) {
            if (bias != null) {
                return x.plus(bias);
            }
            return x;
        }
    }

    @Override
    public @NotNull List<ITensor> parameters() {
        return this.parameters;
    }

    @Override
    public @NotNull List<IModule> subModules() {
        return Collections.emptyList();
    }

    @NotNull
    public ITensor getWeights() {
        return weights;
    }

    @Nullable
    public ITensor getBias() {
        return bias;
    }
}
