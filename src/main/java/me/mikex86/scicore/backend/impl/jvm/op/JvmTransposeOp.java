package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.backend.impl.jvm.JvmTensor;
import me.mikex86.scicore.backend.impl.jvm.JvmDerivedTensor;
import me.mikex86.scicore.op.IUnaryOperation;
import org.jetbrains.annotations.NotNull;

public class JvmTransposeOp implements IUnaryOperation {

    @NotNull
    private final ISciCoreBackend backend;

    public JvmTransposeOp(@NotNull ISciCoreBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull ITensor input) {
        long[] shape = input.getShape();
        if (shape.length > 2) {
            throw new IllegalArgumentException("transpose only supports 2D or lower tensors");
        }
        long[] resultShape = getResultShape(input);

        DataType dataType = input.getDataType();
        ITensor result = new JvmTensor(this.backend, dataType, resultShape);

        long[] resultIndex = new long[resultShape.length];
        long[] inputIndex = new long[shape.length];

        if (dataType.isFloatingPoint()) {
            for (int i = 0; i < resultShape[0]; i++) {
                for (int j = 0; j < resultShape[1]; j++) {
                    resultIndex[0] = i;
                    resultIndex[1] = j;
                    inputIndex[0] = j;
                    inputIndex[1] = i;
                    double inputValue = input.getAsDouble(inputIndex);
                    result.setByDouble(inputValue, resultIndex);
                }
            }
        } else {
            for (int i = 0; i < resultShape[0]; i++) {
                for (int j = 0; j < resultShape[1]; j++) {
                    resultIndex[0] = i;
                    resultIndex[1] = j;
                    inputIndex[0] = j;
                    inputIndex[1] = i;

                    long inputValue = input.getAsLong(inputIndex);
                    result.setByLong(inputValue, resultIndex);
                }
            }
        }
        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull ITensor input) {
        long[] shape = input.getShape();
        if (shape.length > 2) {
            throw new IllegalArgumentException("transpose only supports 2D or lower tensors");
        }
        long[] resultShape = getResultShape(input);

        DataType dataType = input.getDataType();
        return new JvmDerivedTensor(this.backend, resultShape, dataType, () -> perform(input));
    }

    private long @NotNull [] getResultShape(@NotNull ITensor input) {
        long[] shape = input.getShape();
        if (shape.length == 2) {
            return new long[]{shape[1], shape[0]};
        } else if (shape.length == 1) {
            return new long[]{1, shape[0]};
        } else {
            return new long[]{1, 1};
        }
    }
}
