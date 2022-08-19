package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.backend.ITensorImpl;
import me.mikex86.scicore.backend.impl.jvm.JvmDataTensorImpl;
import me.mikex86.scicore.backend.impl.jvm.JvmDerivedTensor;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.op.IUnaryOperation;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

public class JvmExpOp implements IUnaryOperation {

    @NotNull
    private final ISciCoreBackend backend;

    public JvmExpOp(@NotNull ISciCoreBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull ITensor input) {
        long[] shape = input.getShape();
        long nElements = ShapeUtils.getNumElements(shape);
        DataType dataType = input.getDataType();
        ITensorImpl result = new JvmDataTensorImpl(this.backend, dataType, shape);
        if (dataType.isFloatingPoint()) {
            for (long i = 0; i < nElements; i++) {
                double value = input.getAsDoubleFlat(i);
                result.setByDoubleFlat(Math.exp(value), i);
            }
        } else {
            for (long i = 0; i < nElements; i++) {
                long value = input.getAsLongFlat(i);
                result.setByLongFlat((long) Math.exp(value), i);
            }
        }
        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull ITensor input) {
        return new JvmDerivedTensor(backend, input.getShape(), input.getDataType(), () -> perform(input));
    }

}
