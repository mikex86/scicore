package me.mikex86.scicore.backend;

import me.mikex86.scicore.memory.DirectMemoryManager;
import me.mikex86.scicore.graph.IGraphRecorder;
import me.mikex86.scicore.graph.op.IOperation;
import me.mikex86.scicore.graph.OperationType;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.Map;
import java.util.Objects;
import java.util.Optional;

public abstract class AbstractSciCoreBackend implements ISciCoreBackend {

    @Nullable
    private IGraphRecorder operationRecorder;

    @Nullable
    private DirectMemoryManager directMemoryManager;

    public @NotNull Optional<IOperation> getOperation(@NotNull OperationType operationType) {
        IOperation operation = getOperationTable().get(operationType);
        return Optional.ofNullable(operation);
    }

    @NotNull
    protected abstract Map<OperationType, IOperation> getOperationTable();

    public void setOperationRecorder(@NotNull IGraphRecorder operationRecorder) {
        this.operationRecorder = operationRecorder;
    }

    @Override
    public @NotNull IGraphRecorder getOperationRecorder() {
        return Objects.requireNonNull(operationRecorder, "Operation recorder not set");
    }

    public void setDirectMemoryManager(@NotNull DirectMemoryManager directMemoryManager) {
        this.directMemoryManager = directMemoryManager;
    }

    @NotNull
    public DirectMemoryManager getDirectMemoryManager() {
        return Objects.requireNonNull(directMemoryManager, "Direct memory manager not set");
    }
}
