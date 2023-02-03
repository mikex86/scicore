package me.mikex86.scicore.nn.layers;

import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.nn.IModule;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class LayerNorm implements IModule {

    @NotNull
    private final ITensor gamma;

    @NotNull
    private final ITensor beta;

    private final int normalizedShape;

    public LayerNorm(@NotNull ISciCore sciCore, int normalizedShape) {
        this.normalizedShape = normalizedShape;
        this.gamma = sciCore.zeros(DataType.FLOAT32, normalizedShape);
        this.gamma.fill(1f);
        this.beta = sciCore.zeros(DataType.FLOAT32, normalizedShape);
    }

    @Override
    public @NotNull ITensor forward(@NotNull ITensor input) {
        int dim = -1;
        long[] shape = input.getShape();
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] == normalizedShape) {
                dim = i;
                break;
            }
        }
        if (dim == -1) {
            throw new IllegalArgumentException("Input shape " + Arrays.toString(shape) + " does not contain normalized shape " + normalizedShape);
        }
        try (ITensor mean = input.mean(dim, true);
             ITensor variance = input.variance(dim, false, true);
             ITensor varianceEpsilon = variance.plus(1e-5f);
             ITensor std = varianceEpsilon.pow(0.5f);
             ITensor normalized = input.minus(mean).divide(std);
             ITensor rescaled = normalized.multiply(gamma)) {
            return rescaled.plus(beta);
        }
    }

    @Override
    public @NotNull List<ITensor> parameters() {
        return List.of(gamma, beta);
    }

    @Override
    public @NotNull List<IModule> subModules() {
        return Collections.emptyList();
    }
}
