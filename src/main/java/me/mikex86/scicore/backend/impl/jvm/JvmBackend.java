package me.mikex86.scicore.backend.impl.jvm;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.backend.impl.jvm.op.*;
import me.mikex86.scicore.op.*;
import org.jetbrains.annotations.NotNull;

import java.util.List;
import java.util.Map;

public class JvmBackend implements ISciCoreBackend {

    @NotNull
    private final IGraphRecorder operationRecorder = new GraphRecorder(this);

    @NotNull
    private final Map<OperationType, IOperation> operationTable = Map.of(
            OperationType.MATMUL, new JvmMatMulOp(this),
            OperationType.DIVIDED, new JvmDividedOp(this),
            OperationType.PLUS, new JvmPlusOp(this),
            OperationType.REDUCE_SUM, new JvmReduceSumOp(this),
            OperationType.EXP, new JvmExpOp(this),
            OperationType.TRANSPOSE, new JvmTransposeOp(this)
    );

    @Override
    public @NotNull ITensor createTensor(@NotNull DataType dataType, long @NotNull [] shape) {
        return new JvmTensor(this, dataType, shape);
    }

    @Override
    public @NotNull IGraphRecorder getOperationRecorder() {
        return this.operationRecorder;
    }

    public @NotNull IOperation getOperation(@NotNull OperationType operationType) {
        IOperation operation = operationTable.get(operationType);
        if (operation == null) {
            throw new IllegalArgumentException("Operation not implemented for JVMBackend: " + operationType);
        }
        return operation;
    }

    @Override
    public @NotNull ITensor lazyOpTensor(@NotNull IOperation operation, @NotNull List<@NotNull ITensor> inputs) {
        return operation.performLazily(inputs);
    }

    @Override
    public void computeGradients(@NotNull Graph.OperationGraphNode operationNode) {
        OperationType operationType = operationNode.getOperationType();
        IOperation operation = getOperation(operationType);
        if (!(operation instanceof IDifferentiableOperation differentiableOperation)) {
            throw new IllegalStateException("Operation is not differentiable: " + operationType);
        }
        differentiableOperation.computeGradients(operationNode);
    }
}
