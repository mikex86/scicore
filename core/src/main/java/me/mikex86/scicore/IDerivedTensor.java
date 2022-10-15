package me.mikex86.scicore;

import org.jetbrains.annotations.NotNull;

/**
 * A tensor which is derived from another tensor through some operation.
 */
public interface IDerivedTensor extends ITensor {

    @NotNull ITensor result();

}
