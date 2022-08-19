package me.mikex86.scicore.backend;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.op.*;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public interface ISciCoreBackend {

    @NotNull
    ITensorImpl createTensor(@NotNull DataType dataType, long @NotNull [] shape);

    @NotNull IGraphRecorder getOperationRecorder();

    @NotNull IOperation getOperation(@NotNull OperationType operationType);

    @NotNull
    ITensor lazyOpTensor(@NotNull IOperation operation, @NotNull List<@NotNull Object> inputs);

    void computeGradients(@NotNull Graph.OperationGraphNode node);
}
