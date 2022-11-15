package me.mikex86.scicore.nn.act;

import me.mikex86.scicore.nn.IModule;
import me.mikex86.scicore.tensor.ITensor;
import org.jetbrains.annotations.NotNull;

import java.util.Collections;
import java.util.List;

public class Tanh implements IModule {

    @Override
    public @NotNull ITensor forward(@NotNull ITensor input) {
        return input.tanh();
    }

    @Override
    public @NotNull List<IModule> subModules() {
        return Collections.emptyList();
    }
}
