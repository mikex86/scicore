package me.mikex86.scicore.graph;

import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import org.jetbrains.annotations.NotNull;

public interface IGraphRecorder {

    @NotNull ITensor recordOperation(@NotNull OperationType operation, @NotNull OptionBundle optionBundle, @NotNull ITensor... inputs);

    default @NotNull ITensor recordOperation(@NotNull OperationType operation, @NotNull ISciCoreBackend backend, @NotNull ITensor... inputs) {
        return recordOperation(operation, OptionBundle.newEmpty(backend), inputs);
    }

    /**
     * @param sciCoreBackend The backend to use for the graph
     * @param root           the root tensor of the graph
     * @return the graph spanning from the specified root tensor up to all leaves, which root is a function of
     */
    @NotNull Graph getGraphFor(@NotNull ISciCoreBackend sciCoreBackend, @NotNull ITensor root);

    void resetRecording();
}
