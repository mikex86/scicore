package me.mikex86.scicore.backend.impl.cuda.memory;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.utils.Pair;
import org.jetbrains.annotations.NotNull;
import org.lwjgl.system.jemalloc.JEmalloc;

import java.nio.ByteBuffer;

import static jcuda.driver.JCudaDriver.*;
import static me.mikex86.scicore.backend.impl.cuda.Validator.cuCheck;

public class CudaMemoryManager {

    @NotNull
    public CudaMemoryHandle alloc(long size) {
        CUdeviceptr devicePtr = new CUdeviceptr();
        cuCheck(cuMemAlloc(devicePtr, size));
        return new CudaMemoryHandle(devicePtr, size);
    }

    @NotNull
    public CudaMemoryHandle alloc(long nElements, @NotNull DataType dataType) {
        return alloc(dataType.getSizeOf(nElements));
    }

    @NotNull
    public PageLockedMemoryHandle allocPageLocked(long size) {
        Pointer hostPtr = new Pointer();
        cuCheck(cuMemAllocHost(hostPtr, size));
        return new PageLockedMemoryHandle(hostPtr, size);
    }

    @NotNull
    public PageLockedMemoryHandle allocPageLocked(long nElements, @NotNull DataType dataType) {
        return allocPageLocked(dataType.getSizeOf(nElements));
    }

    @NotNull
    public CudaMemoryHandle copyToDevice(@NotNull ITensor tensor) {
        CudaMemoryHandle handle = alloc(tensor.getNumberOfElements(), tensor.getDataType());
        Pair<ByteBuffer, Boolean> pair = tensor.getAsDirectBuffer();
        ByteBuffer directBuffer = pair.getFirst();
        boolean needsFree = pair.getSecond();
        cuCheck(cuMemcpyHtoD(handle.getPointer(), Pointer.to(directBuffer), handle.getSize()));
        if (needsFree) {
            JEmalloc.je_free(directBuffer);
        }
        return handle;
    }
}
