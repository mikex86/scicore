package me.mikex86.scicore.backend.impl.cuda.memory;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import me.mikex86.scicore.backend.impl.cuda.CudaDataContainer;
import me.mikex86.scicore.memory.AbstractMemoryManager;
import me.mikex86.scicore.memory.IMemoryHandle;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.View;
import me.mikex86.scicore.backend.impl.cuda.CudaBackend;
import me.mikex86.scicore.backend.impl.cuda.CudaTensor;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.memory.IMemoryManager;
import me.mikex86.scicore.utils.ViewUtils;
import me.mikex86.scicore.utils.dispose.IDisposable;
import org.jetbrains.annotations.NotNull;

import static jcuda.driver.JCudaDriver.*;
import static me.mikex86.scicore.backend.impl.cuda.Validator.cuCheck;

public class CudaMemoryManager extends AbstractMemoryManager<CudaMemoryHandle> {

    @NotNull
    private final CudaBackend backend;

    public CudaMemoryManager(@NotNull CudaBackend cudaBackend) {
        this.backend = cudaBackend;
    }

    @NotNull
    public CudaMemoryHandle alloc(long size) {
        CUdeviceptr devicePtr = new CUdeviceptr();
        cuCheck(cuMemAlloc(devicePtr, size));
        return new CudaMemoryHandle(this, devicePtr, size);
    }

    @Override
    public @NotNull CudaMemoryHandle calloc(long nBytes) {
        CudaMemoryHandle handle = alloc(nBytes);
        cuCheck(cuMemsetD8(handle.getDevicePointer(), (byte) 0, nBytes));
        registerFinalizer(handle);
        return handle;
    }

    @NotNull
    public CudaMemoryHandle alloc(long nElements, @NotNull DataType dataType) {
        return alloc(dataType.getSizeOf(nElements));
    }

    @Override
    public @NotNull CudaMemoryHandle calloc(long nElements, @NotNull DataType dataType) {
        return calloc(dataType.getSizeOf(nElements));
    }

    @Override
    public void free(@NotNull CudaMemoryHandle memoryHandle) {
        if (memoryHandle.isFreed()) {
            throw new IllegalArgumentException("Handle already freed: " + memoryHandle);
        }
        if (!memoryHandle.canFree()) {
            throw new IllegalArgumentException("Cannot free a sub-handle: " + memoryHandle);
        }
        deactivateFinalizerFor(memoryHandle);
        CUdeviceptr devicePointer = memoryHandle.getDevicePointer();
        cuCheck(cuMemFree(devicePointer));
        memoryHandle.freed = true;
    }

    @Override
    public void copy(@NotNull CudaMemoryHandle dstMemoryHandle, @NotNull CudaMemoryHandle srcMemoryHandle) {
        if (dstMemoryHandle.getSize() != srcMemoryHandle.getSize()) {
            throw new IllegalArgumentException("Source and destination memory handles must be the same size.");
        }
        if (dstMemoryHandle.getDevicePointer().equals(srcMemoryHandle.getDevicePointer())) {
            return;
        }
        if (dstMemoryHandle.getSize() == 0) {
            return;
        }
        cuCheck(cuMemcpyDtoD(dstMemoryHandle.getDevicePointer(), srcMemoryHandle.getDevicePointer(), srcMemoryHandle.getSize()));
    }

    @NotNull
    public CudaMemoryHandle copyToDevice(@NotNull ITensor tensor) {
        CudaMemoryHandle handle = alloc(tensor.getNumberOfElements(), tensor.getDataType());
        DirectMemoryHandle memoryHandle = tensor.getContentsAsDirectMemory();
        cuCheck(cuMemcpyHtoD(handle.getDevicePointer(), Pointer.to(memoryHandle.asByteBuffer()), handle.getSize()));
        if (memoryHandle.canFree()) {
            memoryHandle.free();
        }
        return handle;
    }


    /**
     * Ensures the tensor data is on the device and returns a handle to it.
     * If the tensor is already on the device, a reference handle will be returned. This handle cannot be freed,
     * as the parent handle will be responsible for that. If the tensor is not on the device, memory will be allocated
     * and the data will be copied. The returned handle can be freed.
     *
     * @param tensor the tensor to ensure is on the device.
     * @return a handle to the tensor data on the device.
     */
    @NotNull
    public CudaMemoryHandle ensureOnDevice(@NotNull ITensor tensor) {
        if (tensor instanceof CudaTensor cudaTensor) {
            return cudaTensor.getDataContainer().getDeviceMemoryHandle().createReference();
        } else if (tensor instanceof View view && view.getDataContainer() instanceof CudaDataContainer cudaDataContainer) {
            long offset = cudaDataContainer.getDataType().getSizeOf(view.getOffset());
            return cudaDataContainer.getDeviceMemoryHandle().offset(offset);
        } else {
            return copyToDevice(tensor);
        }
    }

    @Override
    protected @NotNull IDisposable createDisposableFor(@NotNull CudaMemoryHandle memoryHandle) {
        return new CudaMemoryHandleDisposable(memoryHandle.getDevicePointer());
    }

    private record CudaMemoryHandleDisposable(@NotNull CUdeviceptr devicePointer) implements IDisposable {
        @Override
        public void dispose() {
            cuCheck(cuMemFree(devicePointer));
        }
    }
}
