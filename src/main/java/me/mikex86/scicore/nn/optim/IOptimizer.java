package me.mikex86.scicore.nn.optim;

import me.mikex86.scicore.op.IGraph;
import org.jetbrains.annotations.NotNull;

public interface IOptimizer {

    void step(@NotNull IGraph graph);

}
