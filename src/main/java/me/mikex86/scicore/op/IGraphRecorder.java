package me.mikex86.scicore.op;

import me.mikex86.scicore.ITensor;
import org.jetbrains.annotations.NotNull;

public interface IGraphRecorder {

    @NotNull ITensor recordOperation(@NotNull OperationType operation, @NotNull Object... inputs);

    @NotNull Graph finish();

}
