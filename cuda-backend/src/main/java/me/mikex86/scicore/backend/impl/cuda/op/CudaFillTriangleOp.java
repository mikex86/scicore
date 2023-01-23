package me.mikex86.scicore.backend.impl.cuda.op;

import jcuda.Pointer;
import jcuda.driver.CUstream;
import me.mikex86.scicore.backend.impl.cuda.CudaBackend;
import me.mikex86.scicore.backend.impl.cuda.kernel.CudaKernel;
import me.mikex86.scicore.backend.impl.cuda.kernel.CudaKernelLaunchConfig;
import me.mikex86.scicore.backend.impl.cuda.kernel.KernelNameUtility;
import me.mikex86.scicore.backend.impl.cuda.memory.CudaMemoryHandle;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.OptionBundle;
import me.mikex86.scicore.graph.op.IUnaryOperation;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import org.jetbrains.annotations.NotNull;

import java.util.List;

import static jcuda.driver.JCudaDriver.*;
import static me.mikex86.scicore.backend.impl.cuda.Validator.cuCheck;

public class CudaFillTriangleOp implements IUnaryOperation {

    @NotNull
    private final CudaBackend backend;

    public CudaFillTriangleOp(@NotNull CudaBackend backend) {
        this.backend = backend;
    }

    @NotNull
    private final CudaKernel cudaKernel = CudaKernel.loadClassPath("kernels/cuda/fill_triangle.ptx", KernelNameUtility.getForAllDataTypes("fill_triangle", List.of(DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64, DataType.FLOAT32, DataType.FLOAT64)));

    @Override
    public @NotNull ITensor perform(Graph.@NotNull IOperationContext ctx, @NotNull ITensor input) {
        if (input.getShape().length < 2) {
            throw new IllegalArgumentException("Input must be at least 2-dimensional");
        }
        OptionBundle bundle = ctx.getOptionBundle();

        double topValue = bundle.getDouble("topValue").orElseThrow();
        double bottomValue = bundle.getDouble("bottomValue").orElseThrow();

        long[] shape = input.getShape();
        ITensor oneBatchDim = input.view(-1, shape[shape.length - 2], shape[shape.length - 1]);
        long[] oneBatchShape = oneBatchDim.getShape();

        long numBatches = oneBatchShape[0];
        long numRows = oneBatchShape[1];
        long numCols = oneBatchShape[2];

        int numThreadsX = 32;
        int numThreadsY = 32;
        int numThreadsZ = 1;

        int numBlocksX = Math.toIntExact((numRows + numThreadsX - 1) / numThreadsX);
        int numBlocksY = Math.toIntExact((numCols + numThreadsY - 1) / numThreadsY);
        int numBlocksZ = Math.toIntExact(numBatches);

        CudaMemoryHandle cudaMemoryHandle = backend.getCudaMemoryManager().ensureOnDevice(input);

        CUstream stream = new CUstream();
        cuCheck(cuStreamCreate(stream, 0));
        cudaKernel.launchOnStream(stream,
                KernelNameUtility.getForDataType("fill_triangle", oneBatchDim.getDataType()),
                CudaKernelLaunchConfig.builder()
                        .gridDimX(numBlocksX)
                        .gridDimY(numBlocksY)
                        .gridDimZ(numBlocksZ)
                        .blockDimX(numThreadsX)
                        .blockDimY(numThreadsY)
                        .blockDimZ(numThreadsZ)
                        .parameters(
                                Pointer.to(
                                        Pointer.to(cudaMemoryHandle.getDevicePointer()),
                                        Pointer.to(new long[]{numBatches}),
                                        Pointer.to(new long[]{numRows}),
                                        Pointer.to(new long[]{numCols}),
                                        switch (oneBatchDim.getDataType()) {
                                            case INT8 -> Pointer.to(new byte[]{(byte) topValue});
                                            case INT16 -> Pointer.to(new short[]{(short) topValue});
                                            case INT32 -> Pointer.to(new int[]{(int) topValue});
                                            case INT64 -> Pointer.to(new long[]{(long) topValue});
                                            case FLOAT32 -> Pointer.to(new float[]{(float) topValue});
                                            case FLOAT64 -> Pointer.to(new double[]{topValue});
                                            default ->
                                                    throw new IllegalArgumentException("Unsupported data type: " + oneBatchDim.getDataType());
                                        },
                                        switch (oneBatchDim.getDataType()) {
                                            case INT8 -> Pointer.to(new byte[]{(byte) bottomValue});
                                            case INT16 -> Pointer.to(new short[]{(short) bottomValue});
                                            case INT32 -> Pointer.to(new int[]{(int) bottomValue});
                                            case INT64 -> Pointer.to(new long[]{(long) bottomValue});
                                            case FLOAT32 -> Pointer.to(new float[]{(float) bottomValue});
                                            case FLOAT64 -> Pointer.to(new double[]{bottomValue});
                                            default ->
                                                    throw new IllegalArgumentException("Unsupported data type: " + oneBatchDim.getDataType());
                                        }
                                )
                        )
                        .build()
        );
        cuStreamSynchronize(stream);
        cuStreamDestroy(stream);

        return oneBatchDim.view(shape);
    }

    @Override
    public @NotNull ITensor performLazily(Graph.@NotNull IOperationContext ctx, @NotNull ITensor input) {
        return perform(ctx, input);
    }
}
