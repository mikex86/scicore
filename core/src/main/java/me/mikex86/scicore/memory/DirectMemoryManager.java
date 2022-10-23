package me.mikex86.scicore.memory;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import org.jetbrains.annotations.NotNull;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.system.jemalloc.JEmalloc;

public class DirectMemoryManager implements IMemoryManager<DirectMemoryHandle> {

    public static final long NULL = 0;

    @NotNull
    public DirectMemoryHandle alloc(long nBytes) {
        long ptr = JEmalloc.nje_malloc(nBytes);
        if (ptr == NULL) {
            throw new OutOfMemoryError("JEmalloc failed to allocate " + nBytes + " bytes");
        }
        return new DirectMemoryHandle(this, ptr, nBytes);
    }

    @NotNull
    public DirectMemoryHandle calloc(long nBytes) {
        long ptr = JEmalloc.nje_calloc(1, nBytes);
        if (ptr == NULL) {
            throw new OutOfMemoryError("JEmalloc failed to allocate " + nBytes + " bytes");
        }
        return new DirectMemoryHandle(this, ptr, nBytes);
    }

    @NotNull
    public DirectMemoryHandle alloc(long nElements, @NotNull DataType dataType) {
        long nBytes = dataType.getSizeOf(nElements);
        return alloc(nBytes);
    }


    @NotNull
    public DirectMemoryHandle calloc(long nElements, @NotNull DataType dataType) {
        long nBytes = dataType.getSizeOf(nElements);
        return calloc(nBytes);
    }

    @Override
    public void copy(@NotNull DirectMemoryHandle dstMemoryHandle, @NotNull DirectMemoryHandle srcMemoryHandle) {
        if (dstMemoryHandle.getSize() != srcMemoryHandle.getSize()) {
            throw new IllegalArgumentException("Source and destination memory handles must be the same size.");
        }
        if (dstMemoryHandle.getNativePtr() == srcMemoryHandle.getNativePtr()) {
            return;
        }
        if (dstMemoryHandle.getSize() == 0) {
            return;
        }
        MemoryUtil.memCopy(srcMemoryHandle.getNativePtr(), dstMemoryHandle.getNativePtr(), dstMemoryHandle.getSize());
    }

    public void free(@NotNull DirectMemoryHandle directMemoryHandle) {
        if (directMemoryHandle.isFreed()) {
            throw new IllegalArgumentException("Handle already freed: " + directMemoryHandle);
        }
        if (!directMemoryHandle.canFree()) {
            throw new IllegalArgumentException("Cannot free a sub-handle: " + directMemoryHandle);
        }
        JEmalloc.nje_free(directMemoryHandle.getNativePtr());
        directMemoryHandle.freed = true;
    }

    @NotNull
    public DirectMemoryHandle ensureDirect(@NotNull ITensor tensor) {
        return tensor.getContentsAsDirectMemory();
    }
}
