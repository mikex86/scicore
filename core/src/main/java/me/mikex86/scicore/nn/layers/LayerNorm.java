package me.mikex86.scicore.nn.layers;

import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.nn.IModule;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import org.jetbrains.annotations.NotNull;

import java.util.Collections;
import java.util.List;

public class LayerNorm implements IModule {

    @NotNull
    private final ITensor gamma;

    @NotNull
    private final ITensor beta;

    private final int dim;

    public LayerNorm(@NotNull ISciCore sciCore, int dim) {
        this.dim = dim;
        this.gamma = sciCore.scalar(1.0);
        this.beta = sciCore.scalar(0.0);
    }

    @Override
    public @NotNull ITensor forward(@NotNull ITensor input) {
        try (ITensor mean = input.mean(dim);
             ITensor variance = input.variance(dim, false);
             ITensor varianceEpsilon = variance.plus(1e-5);
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
