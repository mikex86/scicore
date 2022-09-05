package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.backend.impl.jvm.JvmTensor;
import me.mikex86.scicore.op.Graph;
import me.mikex86.scicore.op.IDifferentiableSingleParametricOperation;
import me.mikex86.scicore.op.IGraph;
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
        DataType dataType = dataTypeOpt.get();
        JvmTensor result = new JvmTensor(this.backend, dataType, input.getShape());
        long numElements = input.getNumberOfElements();
        // TODO: This is not just horrible, but also slow
        for (int i = 0; i < numElements; i++) {
            result.setByDoubleFlat(input.getAsDoubleFlat(i), i);
        }
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
        return new LazyTensor(backend, input.getShape(), resultDataType, () -> perform(ctx, input, dataTypeOrdinal));
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