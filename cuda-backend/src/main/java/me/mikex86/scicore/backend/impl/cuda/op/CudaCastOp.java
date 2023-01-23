package me.mikex86.scicore.backend.impl.cuda.op;

import jcuda.Pointer;
import me.mikex86.scicore.backend.impl.cuda.CudaBackend;
import me.mikex86.scicore.backend.impl.cuda.CudaTensor;
import me.mikex86.scicore.backend.impl.cuda.kernel.CudaKernel;
import me.mikex86.scicore.backend.impl.cuda.kernel.CudaKernelLaunchConfig;
import me.mikex86.scicore.backend.impl.cuda.kernel.KernelNameUtility;
import me.mikex86.scicore.backend.impl.cuda.memory.CudaMemoryHandle;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.op.IDifferentiableSingleParametricOperation;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.List;
import java.util.Optional;

public class CudaCastOp implements IDifferentiableSingleParametricOperation<Integer> {

    @NotNull
    private final CudaBackend backend;

    @NotNull
    private final CudaKernel kernel = CudaKernel.loadClassPath("kernels/cuda/cast.ptx", KernelNameUtility.getAllTypePermutations("cast", List.of(DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64, DataType.FLOAT32, DataType.FLOAT64)));


    public CudaCastOp(@NotNull CudaBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input, @Nullable Integer dataTypeOrdinal) {
        Validator.validateNotNull(dataTypeOrdinal, "Cannot cast to null data type");
        Optional<DataType> dataTypeOpt = DataType.fromOrdinal(dataTypeOrdinal);
        if (dataTypeOpt.isEmpty()) {
            throw new IllegalArgumentException("Invalid data type ordinal: " + dataTypeOrdinal);
        }
        long[] shape = input.getShape();
        long[] strides = input.getStrides();
        DataType dataType = dataTypeOpt.get();
        try (CudaTensor result = this.backend.createTensor(dataType, shape)) {
            long numElements = input.getNumberOfElements();
            int numThreads = Math.toIntExact(Math.min(numElements, 1024));
            int numBlocks = Math.toIntExact((numElements + numThreads - 1) / numThreads);

            CudaMemoryHandle inputHandle = this.backend.getCudaMemoryManager().ensureOnDevice(input);
            CudaMemoryHandle resultHandle = result.getDataContainer().getDeviceMemoryHandle();

            kernel.launchOnStream(
                    backend.getStream(),
                    KernelNameUtility.getTypePermutation("cast", input.getDataType(), dataType),
                    CudaKernelLaunchConfig.builder()
                            .gridDimX(numBlocks)
                            .blockDimX(numThreads)
                            .parameters(
                                    Pointer.to(
                                            Pointer.to(
                                                    inputHandle.getDevicePointer()
                                            ),
                                            Pointer.to(
                                                    resultHandle.getDevicePointer()
                                            ),
                                            Pointer.to(
                                                    new long[]{numElements}
                                            )
                                    )
                            )
                            .build()
            );
            return result.view(shape, strides);
        }
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input, @Nullable Integer dataTypeOrdinal) {
        Validator.validateNotNull(dataTypeOrdinal, "Cannot cast to null data type");
        Optional<DataType> resultDataTypeOpt = DataType.fromOrdinal(dataTypeOrdinal);
        if (resultDataTypeOpt.isEmpty()) {
            throw new IllegalArgumentException("Invalid data type ordinal: " + dataTypeOrdinal);
        }
        DataType resultDataType = resultDataTypeOpt.get();
        return new LazyTensor(backend, input.getShape(), resultDataType);
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.IDifferentiableNode node, @Nullable Integer dataType) {
        // TODO: Implement
        throw new UnsupportedOperationException("TODO: IMPLEMENT");
    }

    @Override
    public @NotNull Class<Integer> getType() {
        return Integer.class;
    }
}
