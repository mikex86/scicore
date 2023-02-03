package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.op.IBinaryOperation;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import org.jetbrains.annotations.NotNull;

public class GenCpuLessThanOp implements IBinaryOperation {

    @NotNull
    private final GenCPUBackend backend;

    public GenCpuLessThanOp(@NotNull GenCPUBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(Graph.@NotNull IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        if (!a.getDataType().isNumeric()) {
            throw new IllegalArgumentException("A is not numeric");
        }
        if (!b.isScalar() || !b.getDataType().isNumeric()) {
            throw new IllegalArgumentException("B must be a numeric scalar");
        }
        ITensor result = backend.createTensor(DataType.BOOLEAN, a.getShape());
        double bAsDouble = b.elementAsDouble();
        double bAsLong = b.elementAsLong();

        for (long i = 0; i < a.getNumberOfElements(); i++) {
            boolean isLessThan;
            if (a.getDataType().isFloatingPoint()) {
                isLessThan = a.getAsDoubleFlat(i) < bAsDouble;
            } else {
                isLessThan = a.getAsDoubleFlat(i) < bAsLong;
            }
            if (isLessThan) {
                result.setBooleanFlat(true, i);
            }
        }
        result = result.view(a.getShape(), a.getStrides());
        return result;
    }

    @Override
    public @NotNull ITensor performLazily(Graph.@NotNull IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        if (!a.getDataType().isNumeric()) {
            throw new IllegalArgumentException("A is not numeric");
        }
        if (!b.isScalar() || !b.getDataType().isNumeric()) {
            throw new IllegalArgumentException("B must be a numeric scalar");
        }
        return new LazyTensor(backend, a.getShape(), DataType.BOOLEAN);
    }

}
