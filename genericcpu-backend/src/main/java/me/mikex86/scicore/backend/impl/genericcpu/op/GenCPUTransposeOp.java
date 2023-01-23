package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.graph.OptionBundle;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.op.IDifferentiableUnaryOperation;
import me.mikex86.scicore.graph.IGraph;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;

public class GenCPUTransposeOp implements IDifferentiableUnaryOperation {

    @NotNull
    private final GenCPUBackend backend;

    public GenCPUTransposeOp(@NotNull GenCPUBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input) {
        OptionBundle optionBundle = ctx.getOptionBundle();
        int dim1 = optionBundle.getInt("dim1").orElseThrow();
        int dim2 = optionBundle.getInt("dim2").orElseThrow();

        long[] resultShape = getTransposedShape(input, dim1, dim2);
        long[] resultStrides = getTransposedStrides(input, dim1, dim2);
        return input.view(resultShape, resultStrides);
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input) {
        OptionBundle optionBundle = ctx.getOptionBundle();
        int dim1 = optionBundle.getInt("dim1").orElseThrow();
        int dim2 = optionBundle.getInt("dim2").orElseThrow();
        long[] resultShape = getTransposedShape(input, dim1, dim2);

        DataType dataType = input.getDataType();
        return new LazyTensor(this.backend, resultShape, dataType);
    }

    private long @NotNull [] getTransposedShape(@NotNull ITensor input, int dim1, int dim2) {
        long[] shape = input.getShape();
        long[] resultShape = Arrays.copyOf(shape, shape.length);
        resultShape[dim1] = shape[dim2];
        resultShape[dim2] = shape[dim1];
        return resultShape;
    }

    private long @NotNull [] getTransposedStrides(@NotNull ITensor input, int dim1, int dim2) {
        long[] strides = input.getStrides();
        long[] resultStrides = Arrays.copyOf(strides, strides.length);
        resultStrides[dim1] = strides[dim2];
        resultStrides[dim2] = strides[dim1];
        return resultStrides;
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient input) {
        if (input.requiresGradients()) {
            input.accumulateGradient(upstreamGradient.transpose());
        }
    }
}
