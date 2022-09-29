package me.mikex86.scicore.backend.impl.jvm;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.AbstractSciCoreBackend;
import me.mikex86.scicore.backend.impl.jvm.op.*;
import me.mikex86.scicore.op.IOperation;
import me.mikex86.scicore.op.OperationType;
import org.jetbrains.annotations.NotNull;

import java.util.HashMap;
import java.util.Map;

public class JvmBackend extends AbstractSciCoreBackend {

    @NotNull
    private final Map<OperationType, IOperation> operationTable = new HashMap<>();

    {
        operationTable.put(OperationType.MATMUL, new JvmMatMulOp(this));
        operationTable.put(OperationType.DIVIDE, new JvmDivideOp(this));
        operationTable.put(OperationType.PLUS, new JvmPlusOp(this));
        operationTable.put(OperationType.MINUS, new JvmMinusOp(this));
        operationTable.put(OperationType.REDUCE_SUM, new JvmReduceSumOp(this));
        operationTable.put(OperationType.EXP, new JvmExpOp(this));
        operationTable.put(OperationType.TRANSPOSE, new JvmTransposeOp(this));
        operationTable.put(OperationType.POW, new JvmPowerOp(this));
        operationTable.put(OperationType.MULTIPLY, new JvmMultiplyOp(this));
        operationTable.put(OperationType.RELU, new JvmReluOp(this));
        operationTable.put(OperationType.SIGMOID, new JvmSigmoidOp(this));
        operationTable.put(OperationType.ARGMAX, new JvmArgmaxOp(this));
        operationTable.put(OperationType.CAST, new JvmCastOp(this));
    }

    @Override
    public @NotNull ITensor createTensor(@NotNull DataType dataType, long @NotNull [] shape) {
        return new JvmTensor(this, dataType, shape);
    }

    @Override
    protected @NotNull Map<OperationType, IOperation> getOperationTable() {
        return this.operationTable;
    }

}
