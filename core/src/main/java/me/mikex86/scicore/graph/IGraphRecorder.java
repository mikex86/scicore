package me.mikex86.scicore.graph;

import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public interface IGraphRecorder {

    @NotNull ITensor recordOperation(@NotNull OperationType operation, @NotNull OptionBundle optionBundle, @NotNull ITensor... inputs);

    default @NotNull ITensor recordOperation(@NotNull OperationType operation, @NotNull ISciCoreBackend backend, @NotNull ITensor... inputs) {
        return recordOperation(operation, OptionBundle.newEmpty(backend), inputs);
    }

    /**
     * @param sciCoreBackend The backend to use for the graph
     * @param root           the root tensor of the graph
     * @return the graph spanning from the specified root tensor up to all un-computed leaves, which the root is a function of.
     * Already computed tensors will be represented as tensor declarations.
     * @throws IllegalArgumentException if the root tensor was not recorded as an output computed by this graph
     */
    @NotNull Graph getExecutionGraphTo(@NotNull ISciCoreBackend sciCoreBackend, @NotNull ITensor root);

    /**
     * @param sciCoreBackend The backend to use for the graph
     * @param root           the root tensor of the graph
     * @param parameters     all tensors that the graph must span to, even though they may already be computed.
     * @return the graph spanning from the specified root tensor up to all parameters, which the root is a function of.
     * Already computed tensors will be present as operation nodes, but with computed outputs.
     * @throws IllegalArgumentException
     * if the root tensor was not recorded as an output computed by this graph or
     * if root is not a function of all parameters
     */
    @NotNull Graph getBackpropagationGraphTo(@NotNull ISciCoreBackend sciCoreBackend, @NotNull ITensor root, @NotNull List<ITensor> parameters);

    /**
     * Creates a fresh graph to populate with operations.
     */
    void resetRecording();
}
