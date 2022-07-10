package me.mikex86.scicore.op;

import me.mikex86.scicore.ITensor;
import org.jetbrains.annotations.NotNull;

public interface IBinaryOperation extends IOperation {

    @NotNull
    ITensor perform(@NotNull ITensor a, @NotNull ITensor b);

    @NotNull
    ITensor performLazily(@NotNull ITensor a, @NotNull ITensor b);

    void computeGradients(@NotNull IGraph.IDifferentiableNode a, @NotNull IGraph.IDifferentiableNode b);

}
