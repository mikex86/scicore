package me.mikex86.scicore.op;

import me.mikex86.scicore.ITensor;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public interface IOperation {

    @NotNull ITensor perform(@NotNull List<@NotNull ITensor> inputs);

    @NotNull ITensor performLazily(@NotNull List<@NotNull ITensor> inputs);

}
