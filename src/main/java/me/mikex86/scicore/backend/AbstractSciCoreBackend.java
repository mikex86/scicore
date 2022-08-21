package me.mikex86.scicore.backend;

import me.mikex86.scicore.op.GraphRecorder;
import me.mikex86.scicore.op.IGraphRecorder;
import me.mikex86.scicore.op.IOperation;
import me.mikex86.scicore.op.OperationType;
import org.jetbrains.annotations.NotNull;

import java.util.Map;

public abstract class AbstractSciCoreBackend implements ISciCoreBackend {

    @NotNull
    private final IGraphRecorder operationRecorder = new GraphRecorder(this);

    @Override
    public @NotNull IGraphRecorder getOperationRecorder() {
        return this.operationRecorder;
    }

    public @NotNull IOperation getOperation(@NotNull OperationType operationType) {
        IOperation operation = getOperationTable().get(operationType);
        if (operation == null) {
            throw new IllegalArgumentException("Operation not implemented for JVMBackend: " + operationType);
        }
        return operation;
    }

    @NotNull
    protected abstract Map<OperationType, IOperation> getOperationTable();
}
