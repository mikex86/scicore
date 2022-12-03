package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.graph.op.IInplaceOperation;
import me.mikex86.scicore.tensor.ITensor;
import org.jetbrains.annotations.NotNull;

public class GenCPUCopyOp implements IDifferentiableBinaryOperation, IInplaceOperation {

    @Override
    public @NotNull ITensor perform(Graph.@NotNull IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        a.setContents(b);
        return a;
    }

    @Override
    public @NotNull ITensor performLazily(Graph.@NotNull IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        return a;
    }

    @Override
    public void computeGradients(Graph.@NotNull IOperationContext ctx, @NotNull ITensor upstreamGradient, IGraph.@NotNull ITensorNodeWithGradient a, IGraph.@NotNull ITensorNodeWithGradient b) {
        if (b.requiresGradients()) {
            b.accumulateGradient(upstreamGradient);
        }
    }
}
