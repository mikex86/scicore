package me.mikex86.scicore.backend.impl.cuda;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.AbstractSciCoreBackend;
import me.mikex86.scicore.op.IOperation;
import me.mikex86.scicore.op.OperationType;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jetbrains.annotations.NotNull;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.util.HashMap;
import java.util.Map;

import static me.mikex86.scicore.backend.impl.cuda.Validator.cuCheck;
import static org.lwjgl.cuda.CU.*;

public class CudaBackend extends AbstractSciCoreBackend {

    @NotNull
    private static final Logger LOGGER = LogManager.getLogger("CudaBackend");

    @NotNull
    private final Map<OperationType, IOperation> operationTable = new HashMap<>();

    /**
     * Device handle for the main CUDA device used.
     */
    private static final int mainDevice;

    /**
     * Context handle for the main CUDA context used.
     */
    private static final long ctxHandle;

    static {
        cuInit(0);

        try (MemoryStack stack = MemoryStack.stackPush()) {
            int deviceCount;
            {
                IntBuffer deviceCountBuf = stack.mallocInt(1);
                cuCheck(cuDeviceGetCount(deviceCountBuf));
                deviceCount = deviceCountBuf.get(0);
            }
            LOGGER.debug("Num available devices: " + deviceCount);
            int maxComputeCapability = 0;
            int maxComputeCapabilityDevice = 0;
            for (int device = 0; device < deviceCount; device++) {
                ByteBuffer nameBuffer = stack.calloc(256);
                cuCheck(cuDeviceGetName(nameBuffer, device));
                int stringLength = MemoryUtil.memLengthNT1(nameBuffer);
                String deviceName = MemoryUtil.memASCII(nameBuffer, stringLength);
                LOGGER.debug("Device " + device + ": " + deviceName);
                int computeCapability;
                {
                    IntBuffer major = stack.mallocInt(1);
                    IntBuffer minor = stack.mallocInt(1);
                    cuCheck(cuDeviceComputeCapability(major, minor, device));
                    LOGGER.debug("Compute capability: " + major.get(0) + "." + minor.get(0));
                    computeCapability = major.get(0) * 10 + minor.get(0);
                }
                if (computeCapability > maxComputeCapability) {
                    maxComputeCapability = computeCapability;
                    maxComputeCapabilityDevice = device;
                }
            }
            LOGGER.debug("Using device " + maxComputeCapabilityDevice);
            int deviceHandle;
            {
                IntBuffer deviceHandleBuf = stack.mallocInt(1);
                cuCheck(cuDeviceGet(deviceHandleBuf, maxComputeCapabilityDevice));
                deviceHandle = deviceHandleBuf.get(0);
            }
            mainDevice = deviceHandle;

            PointerBuffer ctxHandleBuf = stack.mallocPointer(1);
            cuCheck(cuCtxCreate(ctxHandleBuf, 0, mainDevice));
            ctxHandle = ctxHandleBuf.get(0);
        }
    }

    @Override
    public @NotNull ITensor createTensor(@NotNull DataType dataType, long @NotNull [] shape) {
        return new CudaTensor(this, dataType, shape);
    }

    @Override
    protected @NotNull Map<OperationType, IOperation> getOperationTable() {
        return this.operationTable;
    }
}
