package me.mikex86.scicore.backend.impl.jvm;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.AbstractSciCoreBackend;
import me.mikex86.scicore.backend.impl.jvm.op.*;
import me.mikex86.scicore.op.IOperation;
import me.mikex86.scicore.op.OperationType;
import org.jetbrains.annotations.NotNull;

import java.util.Map;

public class JvmBackend extends AbstractSciCoreBackend {

    @NotNull
    private final Map<OperationType, IOperation> operationTable = Map.of(
            OperationType.MATMUL, new JvmMatMulOp(this),
            OperationType.DIVIDED, new JvmDividedOp(this),
            OperationType.PLUS, new JvmPlusOp(this),
            OperationType.MINUS, new JvmMinusOp(this),
            OperationType.REDUCE_SUM, new JvmReduceSumOp(this),
            OperationType.EXP, new JvmExpOp(this),
            OperationType.TRANSPOSE, new JvmTransposeOp(this),
            OperationType.POW, new JvmPowerOp(this),
            OperationType.MULTIPLY, new JvmMultiplyOp(this)
    );

    @Override
    public @NotNull ITensor createTensor(@NotNull DataType dataType, long @NotNull [] shape) {
        return new JvmTensor(this, dataType, shape);
    }

    @Override
    protected @NotNull Map<OperationType, IOperation> getOperationTable() {
        return this.operationTable;
    }
}
