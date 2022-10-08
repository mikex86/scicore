package me.mikex86.scicore.backend;

import me.mikex86.scicore.backend.except.UnimplementedOperationException;
import me.mikex86.scicore.op.IOperation;
import me.mikex86.scicore.op.OperationType;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jetbrains.annotations.NotNull;

import java.util.*;
import java.util.stream.Collectors;

public class OperationRegistry {

    @NotNull
    private final Deque<ISciCoreBackend> backendStack = new ArrayDeque<>();

    @NotNull
    private final Map<OperationType, IOperation> operationTable = new HashMap<>();

    @NotNull
    private static final Logger LOGGER = LogManager.getLogger("OperationRegistry");

    private boolean fallthrough = true;

    public void pushLayer(@NotNull ISciCoreBackend backend) {
        backendStack.addFirst(backend);
    }

    @NotNull
    public IOperation getOperation(@NotNull OperationType operationType) {
        // lookup cache
        if (operationTable.containsKey(operationType)) {
            return operationTable.get(operationType);
        }

        // fallback
        for (ISciCoreBackend backend : backendStack) {
            Optional<IOperation> operationOpt = backend.getOperation(operationType);
            if (operationOpt.isPresent()) {
                IOperation operation = operationOpt.get();
                operationTable.put(operationType, operation);
                LOGGER.debug("Operation {} found in backend {}", operationType, backend.getClass().getSimpleName());
                return operation;
            } else if (!fallthrough) {
                throw new UnimplementedOperationException("Operation " + operationType + " not found in backend " + backend.getClass().getSimpleName());
            }
        }
        throw new UnimplementedOperationException("Operation not found implemented in any layer. Layers: [ " + backendStack.stream().map(b -> b.getClass().getSimpleName()).collect(Collectors.joining(" ")) + "]");
    }

    public void disableFallthrough() {
        this.fallthrough = false;
    }
}
