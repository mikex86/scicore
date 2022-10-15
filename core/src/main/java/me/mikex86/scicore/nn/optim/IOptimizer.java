package me.mikex86.scicore.nn.optim;

import me.mikex86.scicore.graph.IGraph;
import org.jetbrains.annotations.NotNull;

public interface IOptimizer {

    void step(@NotNull IGraph graph);

}
