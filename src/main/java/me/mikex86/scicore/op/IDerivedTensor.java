package me.mikex86.scicore.op;

import me.mikex86.scicore.ITensor;
import org.jetbrains.annotations.NotNull;

/**
 * A tensor which is derived from another tensor through some operation.
 */
public interface IDerivedTensor extends ITensor {

    @NotNull ITensor result();

}
