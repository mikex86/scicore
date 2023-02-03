package me.mikex86.scicore.backend;

import me.mikex86.scicore.memory.DirectMemoryManager;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.graph.*;
import me.mikex86.scicore.graph.op.IOperation;
import org.jetbrains.annotations.NotNull;

import java.util.Optional;

public interface ISciCoreBackend {

    @NotNull
    ITensor createTensor(@NotNull DataType dataType, long @NotNull [] shape);

    @NotNull Optional<IOperation> getOperation(@NotNull OperationType operationType);

    @NotNull IGraphRecorder getOperationRecorder();

    @NotNull ISciCore.BackendType getBackendType();

    @NotNull DirectMemoryManager getDirectMemoryManager();

    default void synchronize() {
    }

}
