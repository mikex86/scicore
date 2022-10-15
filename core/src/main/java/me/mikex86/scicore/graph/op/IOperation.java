package me.mikex86.scicore.graph.op;

import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.graph.Graph;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public interface IOperation {

    @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull List<@NotNull ITensor> inputs);

    @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull List<@NotNull ITensor> inputs);

}
