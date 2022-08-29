package me.mikex86.scicore.backend.impl.cuda.memory;

import jcuda.Pointer;
import org.jetbrains.annotations.NotNull;

public interface IMemoryHandle {

    void free();

    @NotNull Pointer getPointer();

    long getSize();

    @NotNull IMemoryHandle offset(long offset);


    @NotNull IMemoryHandle createReference();

    /**
     * @return true if the handle is not a reference to another memory handle, which will free the memory. References can be created via {@link #createReference()} and {@link #offset(long)}
     */
    boolean canFree();

}
