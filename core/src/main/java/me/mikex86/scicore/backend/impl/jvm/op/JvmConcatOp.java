package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.OptionBundle;
import me.mikex86.scicore.graph.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import org.jetbrains.annotations.NotNull;


public class JvmConcatOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final ISciCoreBackend backend;

    public JvmConcatOp(@NotNull ISciCoreBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        OptionBundle optionBundle = ctx.getOptionBundle();
        long dimension = optionBundle.getLong("dimension").orElseThrow();
        DataType aDataType = a.getDataType();
        DataType bDataType = b.getDataType();
        if (!aDataType.equals(bDataType)) {
            throw new IllegalArgumentException("Concatenation of tensors with different data types is not supported");
        }
        long[] aShape = a.getShape();
        long[] bShape = b.getShape();
        if (aShape.length != bShape.length) {
            throw new IllegalArgumentException("Tensors must have the same number of dimensions");
        }
        for (int i = 0; i < aShape.length; i++) {
            if (i != dimension && aShape[i] != bShape[i]) {
                throw new IllegalArgumentException("Tensors must have the same shape except for the dimension to concatenate");
            }
        }
        long[] resultShape = new long[aShape.length];
        System.arraycopy(aShape, 0, resultShape, 0, aShape.length);
        resultShape[(int) dimension] += bShape[(int) dimension];
        return new LazyTensor(backend, resultShape, aDataType);
    }

    @Override
    public @NotNull ITensor perform(Graph.@NotNull IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        OptionBundle optionBundle = ctx.getOptionBundle();

        long dimension = optionBundle.getLong("dimension").orElseThrow();

        DataType aDataType = a.getDataType();
        DataType bDataType = b.getDataType();

        if (!aDataType.equals(bDataType)) {
            throw new IllegalArgumentException("Concatenation of tensors with different data types is not supported");
        }

        long[] aShape = a.getShape();
        long[] bShape = b.getShape();

        if (aShape.length != bShape.length) {
            throw new IllegalArgumentException("Tensors must have the same number of dimensions");
        }

        for (int i = 0; i < aShape.length; i++) {
            if (i != dimension && aShape[i] != bShape[i]) {
                throw new IllegalArgumentException("Tensors must have the same shape except for the dimension to concatenate");
            }
        }

        long[] resultShape = new long[aShape.length];
        System.arraycopy(aShape, 0, resultShape, 0, aShape.length);
        resultShape[(int) dimension] += bShape[(int) dimension];

        ITensor result = backend.createTensor(aDataType, resultShape);

        // TODO: Implement

        return result;
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient a, @NotNull IGraph.ITensorNodeWithGradient b) {

    }

}
