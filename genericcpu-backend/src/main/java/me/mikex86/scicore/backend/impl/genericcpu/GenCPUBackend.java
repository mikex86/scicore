package me.mikex86.scicore.backend.impl.genericcpu;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.AbstractSciCoreBackend;
import me.mikex86.scicore.backend.impl.genericcpu.mem.GenCpuMemoryManager;
import me.mikex86.scicore.backend.impl.genericcpu.op.*;
import me.mikex86.scicore.nativelib.LibraryLoader;
import me.mikex86.scicore.op.IOperation;
import me.mikex86.scicore.op.OperationType;
import org.jetbrains.annotations.NotNull;

import java.util.HashMap;
import java.util.Map;

public class GenCPUBackend extends AbstractSciCoreBackend {

    @NotNull
    private final Map<OperationType, IOperation> operationTable = new HashMap<>();

    {
        operationTable.put(OperationType.MATMUL, new GenCPUMatMulOp(this));
        operationTable.put(OperationType.DIVIDED, new GenCPUDividedOp(this));
        operationTable.put(OperationType.PLUS, new GenCPUPlusOp(this));
        operationTable.put(OperationType.MINUS, new GenCPUMinusOp(this));
        operationTable.put(OperationType.REDUCE_SUM, new GenCPUReduceSumOp(this));
        operationTable.put(OperationType.EXP, new GenCPUExpOp(this));
        operationTable.put(OperationType.TRANSPOSE, new GenCPUTransposeOp(this));
        operationTable.put(OperationType.POW, new GenCPUPowerOp(this));
        operationTable.put(OperationType.MULTIPLY, new GenCPUMultiplyOp(this));
        operationTable.put(OperationType.RELU, new GenCPUReluOp(this));
        operationTable.put(OperationType.SIGMOID, new GenCPUSigmoidOp(this));
        operationTable.put(OperationType.ARGMAX, new GenCPUArgmaxOp(this));
    }

    static {
        LibraryLoader.loadLibrary("scicore_genericcpu");
    }

    @NotNull
    private final GenCpuMemoryManager memoryManager = new GenCpuMemoryManager();

    @Override
    public @NotNull ITensor createTensor(@NotNull DataType dataType, long @NotNull [] shape) {
        return new GenCPUTensor(this, dataType, shape);
    }

    @Override
    protected @NotNull Map<OperationType, IOperation> getOperationTable() {
        return this.operationTable;
    }

    @NotNull
    public GenCpuMemoryManager getMemoryManager() {
        return memoryManager;
    }
}
