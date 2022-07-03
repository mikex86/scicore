package me.mikex86.scicore.nn.layers;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.SciCore;
import me.mikex86.scicore.nn.Module;
import org.jetbrains.annotations.NotNull;

public class Linear implements Module {

    @NotNull
    private final SciCore sciCore;

    private final long inputSize;

    @NotNull
    private final ITensor weights;

    public Linear(@NotNull SciCore sciCore, @NotNull DataType dataType, long inputSize, long outputSize) {
        this.sciCore = sciCore;
        this.inputSize = inputSize;
        this.weights = sciCore.random(dataType, outputSize, inputSize);
    }

    @Override
    public @NotNull ITensor forward(@NotNull ITensor input) {
        long[] inputShape = input.getShape();
        if (inputShape.length != 2) {
            throw new IllegalArgumentException("Input must be 2-dimensional for Linear layer");
        }
        if (inputShape[0] != inputSize) {
            throw new IllegalArgumentException("Input size must match the input size of the layer");
        }
        return sciCore.matmul(input, weights);
    }

}
