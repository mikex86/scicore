package me.mikex86.scicore.backend.impl.genericcpu;

import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.backend.AbstractSciCoreBackend;
import me.mikex86.scicore.memory.DirectMemoryManager;
import me.mikex86.scicore.backend.impl.genericcpu.op.*;
import me.mikex86.scicore.nativelib.LibraryLoader;
import me.mikex86.scicore.graph.op.IOperation;
import me.mikex86.scicore.graph.OperationType;
import org.jetbrains.annotations.NotNull;

import java.util.HashMap;
import java.util.Map;

public class GenCPUBackend extends AbstractSciCoreBackend {

    @NotNull
    private final Map<OperationType, IOperation> operationTable = new HashMap<>();

    {
        operationTable.put(OperationType.MATMUL, new GenCPUMatMulOp(this));
        operationTable.put(OperationType.MULTIPLY, new GenCPUMultiplyOp(this));
        operationTable.put(OperationType.PLUS, new GenCPUPlusOp(this));
        operationTable.put(OperationType.MINUS, new GenCPUMinusOp(this));
        operationTable.put(OperationType.DIVIDE, new GenCPUDivideOp(this));
        operationTable.put(OperationType.EXP, new GenCPUExpOp(this));
        operationTable.put(OperationType.RELU, new GenCPUReluOp(this));
        operationTable.put(OperationType.POW, new GenCPUPowOp(this));
        operationTable.put(OperationType.TRANSPOSE, new GenCPUTransposeOp(this));
        operationTable.put(OperationType.CAST, new GenCPUCastOp(this));
        operationTable.put(OperationType.REDUCE_SUM, new GenCPUReduceSumOp(this));
        operationTable.put(OperationType.ARGMAX, new GenCpuArgmaxOp(this));
        operationTable.put(OperationType.PLUS_INPLACE, new GenCPUPlusInplaceOp(this));
        operationTable.put(OperationType.MINUS_INPLACE, new GenCPUMinusInplaceOp(this));
    }

    static {
        LibraryLoader.loadLibrary("scicore_genericcpu");
    }


    @Override
    public @NotNull ITensor createTensor(@NotNull DataType dataType, long @NotNull [] shape) {
        return new GenCPUTensor(this, dataType, shape);
    }

    @NotNull
    @Override
    public ISciCore.BackendType getBackendType() {
        return ISciCore.BackendType.CPU;
    }

    @Override
    protected @NotNull Map<OperationType, IOperation> getOperationTable() {
        return this.operationTable;
    }

    @NotNull
    public DirectMemoryManager getMemoryManager() {
        return getDirectMemoryManager();
    }
}
