package me.mikex86.scicore.backend;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.op.*;
import org.jetbrains.annotations.NotNull;

import java.util.List;
import java.util.Optional;

public interface ISciCoreBackend {

    @NotNull
    ITensor createTensor(@NotNull DataType dataType, long @NotNull [] shape);

    @NotNull Optional<IOperation> getOperation(@NotNull OperationType operationType);

    @NotNull IGraphRecorder getOperationRecorder();

}
