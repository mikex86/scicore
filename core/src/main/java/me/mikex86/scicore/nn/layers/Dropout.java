package me.mikex86.scicore.nn.layers;

import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.nn.IModule;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import org.jetbrains.annotations.NotNull;

import java.util.Collections;
import java.util.List;

public class Dropout implements IModule {

    @NotNull
    private final ISciCore sciCore;

    private final float p;

    public Dropout(@NotNull ISciCore sciCore, float p) {
        this.sciCore = sciCore;
        this.p = p;
    }

    @Override
    public @NotNull ITensor forward(@NotNull ITensor input) {
        ITensor randomValues = sciCore.uniform(DataType.FLOAT32, input.getShape());
        ITensor states = randomValues.lessThan(1f - p).cast(DataType.FLOAT32);
        ITensor inputDropped = input.multiply(states);
        ITensor output;
        if (sciCore.isTraining()) {
            output = inputDropped.divide(1f - p);
        } else {
            output = inputDropped;
        }
        return output;
    }

    @Override
    public @NotNull List<IModule> subModules() {
        return Collections.emptyList();
    }
}
