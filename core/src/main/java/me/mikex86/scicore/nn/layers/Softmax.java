package me.mikex86.scicore.nn.layers;

import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.nn.IModule;
import org.jetbrains.annotations.NotNull;

import java.util.Collections;
import java.util.List;

public class Softmax implements IModule {

    @NotNull
    private final ISciCore sciCore;

    private final int dimension;

    public Softmax(@NotNull ISciCore sciCore, int dimension) {
        this.sciCore = sciCore;
        this.dimension = dimension;
    }

    @Override
    public @NotNull ITensor forward(@NotNull ITensor input) {
        return sciCore.softmax(input, dimension);
    }

    @Override
    public @NotNull List<ITensor> parameters() {
        return Collections.emptyList();
    }

}
