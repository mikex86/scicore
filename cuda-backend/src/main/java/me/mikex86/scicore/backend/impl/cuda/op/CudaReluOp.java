package me.mikex86.scicore.backend.impl.cuda.op;

import jcuda.Pointer;
import me.mikex86.scicore.backend.impl.cuda.CudaBackend;
import me.mikex86.scicore.backend.impl.cuda.CudaTensor;
import me.mikex86.scicore.backend.impl.cuda.codegen.KernelCodeGenerator;
import me.mikex86.scicore.backend.impl.cuda.codegen.UnaryOperationKernelGenerator;
import me.mikex86.scicore.backend.impl.cuda.kernel.CudaKernel;
import me.mikex86.scicore.backend.impl.cuda.kernel.CudaKernelLaunchConfig;
import me.mikex86.scicore.backend.impl.cuda.memory.CudaMemoryHandle;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.op.IDifferentiableUnaryOperation;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public class CudaReluOp implements IDifferentiableUnaryOperation {

    @NotNull
    private final CudaBackend backend;

    public CudaReluOp(@NotNull CudaBackend backend) {
        this.backend = backend;
    }

    @NotNull
    private ITensor relu(@NotNull ITensor input) {
        long[] shape = input.getShape();
        long[] strides = input.getStrides();
        DataType dataType = input.getDataType();

        try (CudaTensor result = backend.createTensor(dataType, shape)) {
            long[] resultShape = result.getShape();
            long nElements = ShapeUtils.getNumElements(resultShape);

            CudaMemoryHandle aDevicePtr = backend.getCudaMemoryManager().ensureOnDevice(input);

            int threadsPerBlock = 1024;
            int nBlocks = Math.toIntExact((nElements + threadsPerBlock - 1) / threadsPerBlock);

            CudaKernel kernel = CudaKernel.jitCompile(
                    KernelCodeGenerator
                            .create()
                            .addFunction(
                                    KernelCodeGenerator.KernelFunction
                                            .builder()
                                            .prefix("extern \"C\" __global__")
                                            .returnType("void")
                                            .functionName("relu_kernel")
                                            .parameter(dataType, 1, "a")
                                            .parameter(dataType, 1, "out")
                                            .body(
                                                    UnaryOperationKernelGenerator
                                                            .builder()
                                                            .shape(shape)
                                                            .operation("a[i] > 0 ? a[i] : 0")
                                                            .build()
                                                            .generateCode()
                                            )
                                            .build()
                            )
                            .buildCode(), List.of("relu_kernel"));
            kernel.launchOnStream(
                    backend.getStream(),
                    "relu_kernel",
                    CudaKernelLaunchConfig.builder()
                            .blockDimX(threadsPerBlock)
                            .gridDimX(nBlocks)
                            .parameters(
                                    Pointer.to(
                                            Pointer.to(aDevicePtr.getDevicePointer()),
                                            Pointer.to(result.getDataContainer().getDeviceMemoryHandle().getDevicePointer())
                                    )
                            )
                            .build()
            );

            return result.view(shape, strides);
        }
    }

    @NotNull
    private ITensor reluGradients(@NotNull ITensor inputTensor) {
        long[] shape = inputTensor.getShape();
        DataType dataType = inputTensor.getDataType();

        CudaTensor result = backend.createTensor(dataType, shape);
        long[] resultShape = result.getShape();
        long nElements = ShapeUtils.getNumElements(resultShape);

        CudaMemoryHandle aDevicePtr = backend.getCudaMemoryManager().ensureOnDevice(inputTensor);

        int threadsPerBlock = 1024;
        int nBlocks = Math.toIntExact((nElements + threadsPerBlock - 1) / threadsPerBlock);

        CudaKernel kernel = CudaKernel.jitCompile(
                KernelCodeGenerator
                        .create()
                        .addFunction(
                                KernelCodeGenerator.KernelFunction
                                        .builder()
                                        .prefix("extern \"C\" __global__")
                                        .returnType("void")
                                        .functionName("relu_gradients_kernel")
                                        .parameter(dataType, 1, "a")
                                        .parameter(dataType, 1, "out")
                                        .body(
                                                UnaryOperationKernelGenerator
                                                        .builder()
                                                        .shape(shape)
                                                        .operation("a[i] > 0 ? 1 : 0")
                                                        .build()
                                                        .generateCode()
                                        )
                                        .build()
                        )
                        .buildCode(), List.of("relu_gradients_kernel"));
        kernel.launchOnStream(
                backend.getStream(),
                "relu_gradients_kernel",
                CudaKernelLaunchConfig.builder()
                        .blockDimX(threadsPerBlock)
                        .gridDimX(nBlocks)
                        .parameters(
                                Pointer.to(
                                        Pointer.to(aDevicePtr.getDevicePointer()),
                                        Pointer.to(result.getDataContainer().getDeviceMemoryHandle().getDevicePointer())
                                )
                        )
                        .build()
        );

        return result;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input) {
        return relu(input);
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input) {
        return new LazyTensor(backend, input.getShape(), input.getDataType());
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient input) {
        if (input.requiresGradients()) {
            ITensor inputTensor = input.getValue();
            try (ITensor gradient = reluGradients(inputTensor)) {
                ITensor finalGradient = gradient.multiply(upstreamGradient);
                input.accumulateGradient(finalGradient);
            }
        }
    }
}
