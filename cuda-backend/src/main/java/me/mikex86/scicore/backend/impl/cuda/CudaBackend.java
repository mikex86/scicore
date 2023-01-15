package me.mikex86.scicore.backend.impl.cuda;

import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.jcublas.cublasHandle;
import me.mikex86.scicore.backend.impl.cuda.op.*;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.backend.AbstractSciCoreBackend;
import me.mikex86.scicore.backend.impl.cuda.memory.CudaMemoryManager;
import me.mikex86.scicore.graph.op.IOperation;
import me.mikex86.scicore.graph.OperationType;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jetbrains.annotations.NotNull;

import java.util.HashMap;
import java.util.Map;

import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.jcublas.JCublas.cublasInit;
import static jcuda.jcublas.JCublas2.cublasCreate;
import static me.mikex86.scicore.backend.impl.cuda.Validator.cuCheck;

public class CudaBackend extends AbstractSciCoreBackend {

    @NotNull
    private static final Logger LOGGER = LogManager.getLogger("CudaBackend");

    @NotNull
    private final Map<OperationType, IOperation> operationTable = new HashMap<>();

    {
        operationTable.put(OperationType.MATMUL, new CudaMatmulOp(this));
        operationTable.put(OperationType.MULTIPLY, new CudaMultiplyOp(this));
        operationTable.put(OperationType.DIVIDE, new CudaDivideOp(this));
        operationTable.put(OperationType.PLUS, new CudaPlusOp(this));
        operationTable.put(OperationType.PLUS_INPLACE, new CudaPlusInplaceOp(this));
        operationTable.put(OperationType.POW, new CudaPowOp(this));
        operationTable.put(OperationType.MINUS, new CudaMinusOp(this));
        operationTable.put(OperationType.MINUS_INPLACE, new CudaMinusInplaceOp(this));
        operationTable.put(OperationType.RESHAPE, new CudaReshapeOp(this));
        operationTable.put(OperationType.CAST, new CudaCastOp(this));
        operationTable.put(OperationType.EXP, new CudaExpOp(this));
        operationTable.put(OperationType.TANH, new CudaTanhOp(this));
        operationTable.put(OperationType.RELU, new CudaReluOp(this));
    }

    @NotNull
    private final CudaMemoryManager cudaMemoryManager = new CudaMemoryManager(this);

    /**
     * Device handle for the main CUDA device used.
     */
    @NotNull
    private static final CUdevice mainDevice;

    /**
     * Context handle for the main CUDA context used.
     */
    @NotNull
    private static final CUcontext ctxHandle;

    @NotNull
    private static final cublasHandle cublasHandle;

    static {
        cuInit(0);

        int deviceCount;
        {
            int[] deviceCountBuf = new int[1];
            cuCheck(cuDeviceGetCount(deviceCountBuf));
            deviceCount = deviceCountBuf[0];
        }
        LOGGER.debug("Num available devices: " + deviceCount);
        int maxComputeCapability = 0;
        int maxComputeCapabilityDeviceOrdinal = 0;
        CUdevice maxComputeCapabilityDevice = null;
        for (int deviceOrdinal = 0; deviceOrdinal < deviceCount; deviceOrdinal++) {
            byte[] nameBuffer = new byte[256];

            CUdevice device = new CUdevice();
            cuCheck(cuDeviceGet(device, deviceOrdinal));
            cuCheck(cuDeviceGetName(nameBuffer, nameBuffer.length, device));

            int strLength = 0;
            while (nameBuffer[strLength] != 0) {
                strLength++;
            }
            String deviceName = new String(nameBuffer, 0, strLength);
            LOGGER.debug("Device " + deviceOrdinal + ": " + deviceName);
            int computeCapability;
            {
                int[] majorBuf = new int[1];
                int[] minorBuf = new int[1];
                cuCheck(cuDeviceGetAttribute(majorBuf, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
                cuCheck(cuDeviceGetAttribute(minorBuf, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
                computeCapability = majorBuf[0] * 10 + minorBuf[0];
            }
            if (computeCapability > maxComputeCapability) {
                maxComputeCapability = computeCapability;
                maxComputeCapabilityDevice = device;
                maxComputeCapabilityDeviceOrdinal = deviceOrdinal;
            }
        }
        if (maxComputeCapabilityDevice == null) {
            throw new IllegalStateException("No CUDA devices found");
        }
        LOGGER.debug("Using device " + maxComputeCapabilityDeviceOrdinal);
        mainDevice = maxComputeCapabilityDevice;

        // Create context
        {
            CUcontext ctx = new CUcontext();
            cuCheck(cuCtxCreate(ctx, 0, mainDevice));
            ctxHandle = ctx;
        }

        {
            // Init cuBLAS
            cuCheck(cublasInit());

            // Init cuBLAS2
            cublasHandle handle = new cublasHandle();
            cuCheck(cublasCreate(handle));
            cublasHandle = handle;
        }
    }

    @Override
    public @NotNull CudaTensor createTensor(@NotNull DataType dataType, long @NotNull [] shape) {
        return new CudaTensor(this, dataType, shape);
    }

    @NotNull
    @Override
    public ISciCore.BackendType getBackendType() {
        return ISciCore.BackendType.CUDA;
    }

    @Override
    protected @NotNull Map<OperationType, IOperation> getOperationTable() {
        return this.operationTable;
    }

    @NotNull
    public CUcontext getCtxHandle() {
        return ctxHandle;
    }

    @NotNull
    public CudaMemoryManager getCudaMemoryManager() {
        return cudaMemoryManager;
    }

    @NotNull
    public static cublasHandle getCublasHandle() {
        return cublasHandle;
    }
}
