package me.mikex86.scicore.backend.impl.cuda.memory;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.utils.Pair;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;
import org.lwjgl.system.jemalloc.JEmalloc;

import java.nio.ByteBuffer;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static me.mikex86.scicore.backend.impl.cuda.Validator.cuCheck;

public class CudaMemoryManager {

    @NotNull
    public MemoryHandle alloc(long size) {
        CUdeviceptr devicePtr = new CUdeviceptr();
        cuCheck(cuMemAlloc(devicePtr, size));
        return new MemoryHandle(devicePtr, size);
    }

    @NotNull
    public MemoryHandle alloc(long nElements, @NotNull DataType dataType) {
        return alloc(dataType.getSizeOf(nElements));
    }

    @NotNull
    public MemoryHandle copyToDevice(@NotNull ITensor tensor) {
        MemoryHandle handle = alloc(tensor.getNumberOfElements(), tensor.getDataType());
        Pair<ByteBuffer, Boolean> pair = tensor.getAsDirectBuffer();
        ByteBuffer directBuffer = pair.getFirst();
        boolean needsFree = pair.getSecond();
        cuCheck(cuMemcpyHtoD(handle.getDevicePtr(), Pointer.to(directBuffer), handle.getSize()));
        if (needsFree) {
            JEmalloc.je_free(directBuffer);
        }
        return handle;
    }

}
