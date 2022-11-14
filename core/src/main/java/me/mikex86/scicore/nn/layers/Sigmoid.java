package me.mikex86.scicore.nn.layers;

import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.nn.IModule;
import org.jetbrains.annotations.NotNull;

import java.util.Collections;
import java.util.List;

public class Sigmoid implements IModule {

    @Override
    public @NotNull ITensor forward(@NotNull ITensor input) {
        return input.sigmoid();
    }

    @Override
    public @NotNull List<IModule> subModules() {
        return Collections.emptyList();
    }

}
