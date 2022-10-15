package me.mikex86.scicore.graph.op;

import me.mikex86.scicore.graph.Graph;
import org.jetbrains.annotations.NotNull;

public interface IDifferentiableOperation {

    void computeGradients(@NotNull Graph.OperationGraphNode operationNode);

}
