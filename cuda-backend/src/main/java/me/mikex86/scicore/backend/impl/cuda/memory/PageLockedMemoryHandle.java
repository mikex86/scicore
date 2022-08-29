package me.mikex86.scicore.backend.impl.cuda.memory;

import jcuda.Pointer;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import static jcuda.driver.JCudaDriver.cuMemFreeHost;
import static me.mikex86.scicore.backend.impl.cuda.Validator.cuCheck;

public class PageLockedMemoryHandle implements IMemoryHandle {

    @Nullable
    private final PageLockedMemoryHandle parent;

    @NotNull
    private final Pointer pointer;

    private final long size;

    private boolean freed = false;

    public PageLockedMemoryHandle(@NotNull Pointer pointer, long size) {
        this.parent = null;
        this.pointer = pointer;
        this.size = size;
    }

    private PageLockedMemoryHandle(@NotNull PageLockedMemoryHandle parent, long offset) {
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
        cuCheck(cuMemFreeHost(pointer));
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
    public PageLockedMemoryHandle offset(long offset) {
        if (offset < 0) {
            throw new IllegalArgumentException("offset must be >= 0");
        }
        return new PageLockedMemoryHandle(this, offset);
    }

    @Override
    public @NotNull PageLockedMemoryHandle createReference() {
        return new PageLockedMemoryHandle(this, 0);
    }

    @Override
    public boolean canFree() {
        return this.parent == null;
    }
}
