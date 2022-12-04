package me.mikex86.scicore.backend.impl.jvm;

import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.backend.AbstractSciCoreBackend;
import me.mikex86.scicore.backend.impl.jvm.op.*;
import me.mikex86.scicore.graph.op.IOperation;
import me.mikex86.scicore.graph.OperationType;
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
        operationTable.put(OperationType.RESHAPE, new JvmReshapeOp(this));
        operationTable.put(OperationType.POW, new JvmPowOp(this));
        operationTable.put(OperationType.MULTIPLY, new JvmMultiplyOp(this));
        operationTable.put(OperationType.RELU, new JvmReluOp(this));
        operationTable.put(OperationType.SIGMOID, new JvmSigmoidOp(this));
        operationTable.put(OperationType.TANH, new JvmTanhOp(this));
        operationTable.put(OperationType.ARGMAX, new JvmArgmaxOp(this));
        operationTable.put(OperationType.CAST, new JvmCastOp(this));
        operationTable.put(OperationType.ONE_HOT, new JvmOneHotOp(this));
        operationTable.put(OperationType.GET, new JvmGetOp(this));
        operationTable.put(OperationType.PLUS_INPLACE, new JvmPlusInplaceOp(this));
        operationTable.put(OperationType.MINUS_INPLACE, new JvmMinusInplaceOp(this));
        operationTable.put(OperationType.COMPARE_ELEMENTS, new JvmCompareElementsOp(this));
        operationTable.put(OperationType.CONCAT, new JvmConcatOp(this));
        operationTable.put(OperationType.STACK, new JvmStackOp(this));
    }

    @Override
    public @NotNull ITensor createTensor(@NotNull DataType dataType, long @NotNull [] shape) {
        return new JvmTensor(this, dataType, shape);
    }

    @NotNull
    @Override
    public ISciCore.BackendType getBackendType() {
        return ISciCore.BackendType.JVM;
    }

    @Override
    protected @NotNull Map<OperationType, IOperation> getOperationTable() {
        return this.operationTable;
    }

}
