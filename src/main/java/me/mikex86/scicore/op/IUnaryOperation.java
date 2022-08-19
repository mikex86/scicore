package me.mikex86.scicore.op;

import me.mikex86.scicore.ITensor;
import org.jetbrains.annotations.NotNull;

public interface IUnaryOperation extends IOperation {

    @NotNull ITensor perform(@NotNull ITensor input);

    @NotNull ITensor performLazily(@NotNull ITensor input);

}
