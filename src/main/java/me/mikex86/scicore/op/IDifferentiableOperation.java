package me.mikex86.scicore.op;

import org.jetbrains.annotations.NotNull;

public interface IDifferentiableOperation {

    void computeGradients(@NotNull Graph.OperationGraphNode operationNode);

}
