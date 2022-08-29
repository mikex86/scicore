package me.mikex86.scicore.backend.impl.cuda.memory;

import jcuda.driver.CUdeviceptr;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import static jcuda.driver.JCudaDriver.cuMemFree;
import static me.mikex86.scicore.backend.impl.cuda.Validator.cuCheck;

/**
 * Garbage collected cuda device memory handle
 */
public class MemoryHandle {

    @Nullable
    private final MemoryHandle parent;

    @NotNull
    private final CUdeviceptr devicePtr;

    private final long size;

    private boolean freed = false;

    public MemoryHandle(@NotNull CUdeviceptr devicePtr, long size) {
        this.parent = null;
        this.devicePtr = devicePtr;
        this.size = size;
    }

    private MemoryHandle(@NotNull MemoryHandle parent, long offset) {
        this.parent = parent;
        this.devicePtr = parent.devicePtr.withByteOffset(offset);
        this.size = parent.size - offset;
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        free();
    }

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
    public CUdeviceptr getDevicePtr() {
        return devicePtr;
    }

    public long getSize() {
        return size;
    }

    @NotNull
    public MemoryHandle offset(long offset) {
        return new MemoryHandle(this, offset);
    }
}
