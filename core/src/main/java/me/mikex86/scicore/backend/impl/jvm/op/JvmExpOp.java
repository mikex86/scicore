package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.op.IDifferentiableUnaryOperation;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

public class JvmExpOp implements IDifferentiableUnaryOperation {

    @NotNull
    private final JvmBackend backend;

    public JvmExpOp(@NotNull JvmBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input) {
        long[] shape = input.getShape();
        long[] strides = input.getStrides();
        long nElements = ShapeUtils.getNumElements(shape);
        DataType dataType = input.getDataType();
        ITensor result = backend.createTensor(dataType, shape);
        for (long i = 0; i < nElements; i++) {
            double value = input.getAsDoubleFlat(i);
            result.setByDoubleFlat(Math.exp(value), i);
        }
        result = result.getReshapedView(shape, strides);
        ctx.saveForBackward("exp", result);
        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input) {
        return new LazyTensor(backend, input.getShape(), input.getDataType());
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient input) {
        if (input.requiresGradients()) {
            ITensor exp = ctx.getSavedTensor("exp").orElseThrow(() -> new IllegalStateException("No saved tensor named \"exp\" found"));
            input.accumulateGradient(upstreamGradient.multiply(exp));
        }
    }
}
