package me.mikex86.scicore.op;

import me.mikex86.scicore.ITensor;
import org.jetbrains.annotations.NotNull;

public interface IBiParametricOperation<F, S> extends IParametricOperation {

    @NotNull ITensor perform(@NotNull ITensor tensor, F f, S s);

    @NotNull ITensor performLazily(@NotNull ITensor tensor, F f, S s);

    void computeGradient(@NotNull IGraph.IDifferentiableNode tensor, F f, S s);

}
