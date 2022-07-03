package me.mikex86.scicore.nn.layers;

import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.SciCore;
import me.mikex86.scicore.nn.Module;
import org.jetbrains.annotations.NotNull;

public class Softmax implements Module {

    @NotNull
    private final SciCore sciCore;

    private final int dimension;

    public Softmax(@NotNull SciCore sciCore, int dimension) {
        this.sciCore = sciCore;
        this.dimension = dimension;
    }

    @Override
    public @NotNull ITensor forward(@NotNull ITensor input) {
        return sciCore.softmax(input, dimension);
    }

}
