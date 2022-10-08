package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.op.Graph;
import me.mikex86.scicore.op.IDifferentiableUnaryOperation;
import me.mikex86.scicore.op.IGraph;
import org.jetbrains.annotations.NotNull;

public class GenCPUTransposeOp implements IDifferentiableUnaryOperation {

    @NotNull
    private final GenCPUBackend backend;

    public GenCPUTransposeOp(@NotNull GenCPUBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input) {
        long[] shape = input.getShape();
        if (shape.length > 2) {
            throw new IllegalArgumentException("transpose only supports 2D or lower tensors");
        }
        long[] newShape = getResultShape(input);
        long[] strides = getTransposedStrides(input);
        return input.getReshapedView(newShape, strides);
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input) {
        long[] shape = input.getShape();
        if (shape.length > 2) {
            throw new IllegalArgumentException("transpose only supports 2D or lower tensors");
        }
        long[] resultShape = getResultShape(input);

        DataType dataType = input.getDataType();
        return new LazyTensor(this.backend, resultShape, dataType, () -> perform(ctx, input));
    }

    private long @NotNull [] getResultShape(@NotNull ITensor input) {
        long[] shape = input.getShape();
        if (shape.length == 2) {
            return new long[]{shape[1], shape[0]};
        } else if (shape.length == 1) {
            return new long[]{1, shape[0]};
        } else {
            throw new IllegalArgumentException("transpose only supports 2D or lower tensors");
        }
    }

    private long @NotNull [] getTransposedStrides(@NotNull ITensor input) {
        long[] strides = input.getStrides();
        if (strides.length == 2) {
            return new long[]{strides[1], strides[0]};
        } else if (strides.length == 1) {
            return new long[]{1, 1};
        } else {
            throw new IllegalArgumentException("transpose only supports 2D or lower tensors");
        }
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient input) {
        if (input.requiresGradients()) {
            input.accumulateGradient(upstreamGradient.transpose());
        }
    }
}
