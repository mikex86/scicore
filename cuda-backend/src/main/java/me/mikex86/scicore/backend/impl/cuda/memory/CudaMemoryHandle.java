package me.mikex86.scicore.backend.impl.cuda.memory;

import jcuda.driver.CUdeviceptr;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import static jcuda.driver.JCudaDriver.cuMemFree;
import static me.mikex86.scicore.backend.impl.cuda.Validator.cuCheck;

/**
 * Garbage collected cuda device memory handle
 */
public class CudaMemoryHandle implements IMemoryHandle {

    @Nullable
    private final CudaMemoryHandle parent;

    @NotNull
    private final CUdeviceptr devicePtr;

    private final long size;

    private boolean freed = false;

    public CudaMemoryHandle(@NotNull CUdeviceptr devicePtr, long size) {
        this.parent = null;
        this.devicePtr = devicePtr;
        this.size = size;
    }

    private CudaMemoryHandle(@NotNull CudaMemoryHandle parent, long offset) {
        this.parent = parent;
        this.devicePtr = parent.devicePtr.withByteOffset(offset);
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
        cuCheck(cuMemFree(devicePtr));
        freed = true;
    }

    @NotNull
    @Override
    public CUdeviceptr getPointer() {
        return devicePtr;
    }

    @Override
    public long getSize() {
        return size;
    }

    @NotNull
    public CudaMemoryHandle offset(long offset) {
        if (offset < 0) {
            throw new IllegalArgumentException("offset must be >= 0");
        }
        return new CudaMemoryHandle(this, offset);
    }

    @Override
    public @NotNull CudaMemoryHandle createReference() {
        return new CudaMemoryHandle(this, 0);
    }

    @Override
    public boolean canFree() {
        return parent == null;
    }
}
