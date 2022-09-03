package me.mikex86.scicore.op;

import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import org.jetbrains.annotations.NotNull;

public interface IGraphRecorder {

    @NotNull ITensor recordOperation(@NotNull OperationType operation, @NotNull ITensor... inputs);

    /**
     * @param sciCoreBackend
     * @param root           the root tensor of the graph
     * @return the graph spanning from the specified root tensor up to all leaves, which root is a function of
     */
    @NotNull Graph getGraphFor(@NotNull ISciCoreBackend sciCoreBackend, @NotNull ITensor root);

    void resetRecording();
}
