package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.op.IDifferentiableSingleParametricOperation;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.Optional;

public class JvmCastOp implements IDifferentiableSingleParametricOperation<Integer> {

    @NotNull
    private final JvmBackend backend;

    public JvmCastOp(@NotNull JvmBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input, @Nullable Integer dataTypeOrdinal) {
        Validator.validateNotNull(dataTypeOrdinal, "Cannot cast to null data type");
        Optional<DataType> dataTypeOpt = DataType.fromOrdinal(dataTypeOrdinal);
        if (dataTypeOpt.isEmpty()) {
            throw new IllegalArgumentException("Invalid data type ordinal: " + dataTypeOrdinal);
        }
        long[] shape = input.getShape();
        long[] strides = input.getStrides();
        DataType dataType = dataTypeOpt.get();
        ITensor result = this.backend.createTensor(dataType, shape);
        long numElements = input.getNumberOfElements();
        // TODO: This is not just horrible, but also slow
        for (int i = 0; i < numElements; i++) {
            result.setByDoubleFlat(input.getAsDoubleFlat(i), i);
        }
        result = result.getReshapedView(shape, strides);
        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input, @Nullable Integer dataTypeOrdinal) {
        Validator.validateNotNull(dataTypeOrdinal, "Cannot cast to null data type");
        Optional<DataType> resultDataTypeOpt = DataType.fromOrdinal(dataTypeOrdinal);
        if (resultDataTypeOpt.isEmpty()) {
            throw new IllegalArgumentException("Invalid data type ordinal: " + dataTypeOrdinal);
        }
        DataType resultDataType = resultDataTypeOpt.get();
        return new LazyTensor(backend, input.getShape(), resultDataType);
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.IDifferentiableNode node, @Nullable Integer dataType) {
        // TODO: Implement
        throw new UnsupportedOperationException("TODO: IMPLEMENT");
    }

    @Override
    public @NotNull Class<Integer> getType() {
        return Integer.class;
    }
}
