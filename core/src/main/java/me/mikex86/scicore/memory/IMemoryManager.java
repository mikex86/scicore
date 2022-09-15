package me.mikex86.scicore.memory;

import me.mikex86.scicore.DataType;
import org.jetbrains.annotations.NotNull;

public interface IMemoryManager<T extends IMemoryHandle> {

    @NotNull
    T alloc(long nBytes);

    @NotNull
    T calloc(long nBytes);

    @NotNull
    T alloc(long nElements, @NotNull DataType dataType);

    @NotNull
    T calloc(long nElements, @NotNull DataType dataType);



    /**
     * Copies the data from the source memory handle to the destination memory handle.
     * @param dstMemoryHandle the destination memory handle.
     * @param srcMemoryHandle the source memory handle.
     */
    void copy(@NotNull T dstMemoryHandle, @NotNull T srcMemoryHandle);
}
