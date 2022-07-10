package me.mikex86.scicore.backend;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.op.IOperation;
import me.mikex86.scicore.op.IGraphRecorder;
import me.mikex86.scicore.op.OperationType;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public interface ISciCoreBackend {

    @NotNull
    ITensorImpl createTensor(@NotNull DataType dataType, long @NotNull [] shape);

    @NotNull IGraphRecorder getOperationRecorder();

    @NotNull IOperation getOperation(@NotNull OperationType operationType);

    @NotNull
    ITensor lazyOpTensor(@NotNull IOperation operation, @NotNull List<@NotNull Object> inputs);

    void computeGradients(@NotNull IOperation operation, @NotNull List<IGraph.IGraphNode> inputs);
}
