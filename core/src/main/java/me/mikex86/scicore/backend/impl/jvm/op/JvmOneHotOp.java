package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.op.ISingleParametricOperation;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

public class JvmOneHotOp implements ISingleParametricOperation<Long> {

    @NotNull
    private final JvmBackend backend;

    public JvmOneHotOp(@NotNull JvmBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input, @Nullable Long nClasses) {
        if (nClasses == null) {
            throw new IllegalArgumentException("nClasses must not be null");
        }
        DataType dataType = input.getDataType();
        if (!dataType.isInteger()) {
            throw new IllegalArgumentException("One-hot encoding requires indices to be integer type");
        }
        long[] shape = input.getShape();
        long[] resultShape = new long[shape.length + 1];
        System.arraycopy(shape, 0, resultShape, 0, shape.length);
        resultShape[shape.length] = nClasses;

        ITensor oneHotTensor = backend.createTensor(dataType, resultShape);
        long[] index = new long[shape.length];
        long[] resultIndex = new long[shape.length + 1];
        do {
            long indexValue = input.getAsLong(index);
            if (indexValue < 0 || indexValue >= nClasses) {
                throw new IllegalArgumentException("Index " + indexValue + " is out of bounds for one-hot encoding with " + nClasses + " classes");
            }
            System.arraycopy(index, 0, resultIndex, 0, index.length);
            resultIndex[index.length] = indexValue;
            oneHotTensor.setByLong(1, resultIndex);
        } while (ShapeUtils.incrementIndex(index, shape));
        return oneHotTensor;
    }

    @Override
    public @NotNull ITensor performLazily(Graph.@NotNull IOperationContext ctx, @NotNull ITensor input, @Nullable Long nClasses) {
        if (nClasses == null) {
            throw new IllegalArgumentException("nClasses must not be null");
        }
        DataType dataType = input.getDataType();
        if (!dataType.isInteger()) {
            throw new IllegalArgumentException("One-hot encoding requires indices to be integer type");
        }
        long[] shape = input.getShape();
        long[] resultShape = new long[shape.length + 1];
        System.arraycopy(shape, 0, resultShape, 0, shape.length);
        resultShape[shape.length] = nClasses;
        return new LazyTensor(backend, resultShape, dataType);
    }

    @Override
    public @NotNull Class<Long> getType() {
        return Long.class;
    }
}
