package me.mikex86.scicore.backend.impl.cuda.memory;

import jcuda.driver.CUdeviceptr;
import me.mikex86.scicore.memory.IMemoryHandle;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import static jcuda.driver.JCudaDriver.cuMemFree;
import static me.mikex86.scicore.backend.impl.cuda.Validator.cuCheck;

/**
 * Garbage collected cuda device memory handle
 */
public class CudaMemoryHandle implements IMemoryHandle<CudaMemoryHandle> {

    @Nullable
    private final CudaMemoryHandle parent;

    @NotNull
    private final CUdeviceptr devicePtr;

    private final long size;

    boolean freed = false;

    @NotNull
    private final CudaMemoryManager memoryManager;

    public CudaMemoryHandle(@NotNull CudaMemoryManager memoryManager, @NotNull CUdeviceptr devicePtr, long size) {
        this.memoryManager = memoryManager;
        this.parent = null;
        this.devicePtr = devicePtr;
        this.size = size;
    }

    private CudaMemoryHandle(@NotNull CudaMemoryHandle parent, long offset) {
        this.memoryManager = parent.memoryManager;
        this.parent = parent;
        this.devicePtr = parent.devicePtr.withByteOffset(offset);
        this.size = parent.size - offset;
    }

    @Override
    public void free() {
        if (freed) {
            throw new IllegalArgumentException("Memory already freed: " + this);
        }
        if (parent != null) {
            throw new IllegalArgumentException("Cannot free a sub-handle: " + this);
        }
        this.memoryManager.free(this);
    }

    @NotNull
    public CUdeviceptr getDevicePointer() {
        return devicePtr;
    }

    @Override
    public long getSize() {
        return size;
    }

    /**
     * Creates a reference handle to a subregion of this handle. The parent handle will be responsible for freeing the
     * memory.
     * @param offset the offset of the subregion
     * @return a reference handle to the subregion
     */
    @NotNull
    public CudaMemoryHandle offset(long offset) {
        if (offset < 0) {
            throw new IllegalArgumentException("offset must be >= 0");
        }
        if (offset >= size) {
            throw new IllegalArgumentException("offset must be < size");
        }
        return new CudaMemoryHandle(this, offset);
    }

    @Override
    public @NotNull CudaMemoryHandle offset(long offset, long size) {
        if (offset < 0) {
            throw new IllegalArgumentException("offset must be >= 0");
        }
        if (offset >= this.size) {
            throw new IllegalArgumentException("offset must be < size");
        }
        if (size < 0) {
            throw new IllegalArgumentException("size must be >= 0");
        }
        if (offset + size > this.size) {
            throw new IllegalArgumentException("offset + size must be <= size");
        }
        return new CudaMemoryHandle(this, offset);
    }

    @Override
    public @Nullable CudaMemoryHandle getParent() {
        return parent;
    }

    @Override
    public boolean isFreed() {
        return freed;
    }
}
