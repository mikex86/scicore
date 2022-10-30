package me.mikex86.scicore.nn.optim;

import me.mikex86.scicore.ITensor;
import org.jetbrains.annotations.NotNull;

public interface IOptimizer {

    void step(@NotNull ITensor loss);

}
