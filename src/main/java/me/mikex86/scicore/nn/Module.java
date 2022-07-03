package me.mikex86.scicore.nn;

import me.mikex86.scicore.ITensor;
import org.jetbrains.annotations.NotNull;

public interface Module {

    @NotNull
    ITensor forward(@NotNull ITensor input);

}
