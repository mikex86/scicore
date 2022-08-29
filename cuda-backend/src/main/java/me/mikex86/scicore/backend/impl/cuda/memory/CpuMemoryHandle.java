package me.mikex86.scicore.backend.impl.cuda.memory;

import jcuda.Pointer;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.lwjgl.system.jemalloc.JEmalloc;

public class CpuMemoryHandle implements IMemoryHandle {

    @Nullable
    private final CpuMemoryHandle parent;

    @NotNull
    private final Pointer pointer;

    private final long size;

    private boolean freed = false;

    public CpuMemoryHandle(@NotNull Pointer pointer, long size) {
        this.parent = null;
        this.pointer = pointer;
        this.size = size;
    }

    private CpuMemoryHandle(@NotNull CpuMemoryHandle parent, long offset) {
        this.parent = parent;
        this.pointer = parent.getPointer().withByteOffset(offset);
        this.size = parent.size - offset;
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        free();
    }

    @Override
    public void free() {
        if (freed) {
            return;
        }
        if (parent != null) {
            return; // parent will free
        }
        JEmalloc.je_free(pointer.getByteBuffer());
        freed = true;
    }

    @NotNull
    @Override
    public Pointer getPointer() {
        return pointer;
    }

    @Override
    public long getSize() {
        return size;
    }

    @NotNull
    public CpuMemoryHandle offset(long offset) {
        if (offset < 0) {
            throw new IllegalArgumentException("offset must be >= 0");
        }
        return new CpuMemoryHandle(this, offset);
    }

    @Override
    public @NotNull CpuMemoryHandle createReference() {
        return new CpuMemoryHandle(this, 0);
    }

    @Override
    public boolean canFree() {
        return this.parent == null;
    }
}
