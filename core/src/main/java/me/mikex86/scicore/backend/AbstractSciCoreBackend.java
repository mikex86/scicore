package me.mikex86.scicore.backend;

import me.mikex86.scicore.op.IGraphRecorder;
import me.mikex86.scicore.op.IOperation;
import me.mikex86.scicore.op.OperationType;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.Map;
import java.util.Objects;
import java.util.Optional;

public abstract class AbstractSciCoreBackend implements ISciCoreBackend {

    @Nullable
    private IGraphRecorder operationRecorder;

    public @NotNull Optional<IOperation> getOperation(@NotNull OperationType operationType) {
        IOperation operation = getOperationTable().get(operationType);
        return Optional.ofNullable(operation);
    }

    @NotNull
    protected abstract Map<OperationType, IOperation> getOperationTable();

    @Override
    public @NotNull IGraphRecorder getOperationRecorder() {
        return Objects.requireNonNull(operationRecorder, "Operation recorder not set");
    }

    public void setOperationRecorder(@Nullable IGraphRecorder operationRecorder) {
        this.operationRecorder = operationRecorder;
    }
}
