package me.mikex86.scicore.backend.impl.cuda.op;

import jcuda.Pointer;
import me.mikex86.scicore.backend.impl.cuda.CudaBackend;
import me.mikex86.scicore.backend.impl.cuda.CudaTensor;
import me.mikex86.scicore.backend.impl.cuda.codegen.BroadcastingElementWiseOperationKernelCodeGenerator;
import me.mikex86.scicore.backend.impl.cuda.codegen.KernelCodeGenerator;
import me.mikex86.scicore.backend.impl.cuda.codegen.UnaryOperationKernelGenerator;
import me.mikex86.scicore.backend.impl.cuda.kernel.CudaKernel;
import me.mikex86.scicore.backend.impl.cuda.kernel.CudaKernelLaunchConfig;
import me.mikex86.scicore.backend.impl.cuda.memory.CudaMemoryHandle;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.graph.op.IDifferentiableUnaryOperation;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.utils.GradientUtil;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.util.List;
import java.util.Optional;

public class CudaExpOp implements IDifferentiableUnaryOperation {

    @NotNull
    private final CudaBackend backend;

    public CudaExpOp(@NotNull CudaBackend backend) {
        this.backend = backend;
    }

    @NotNull
    private ITensor exp(@NotNull ITensor input) {
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
                                            .functionName("exp_kernel")
                                            .parameter(dataType, 1, "a")
                                            .parameter(dataType, 1, "out")
                                            .body(
                                                    UnaryOperationKernelGenerator
                                                            .builder()
                                                            .shape(shape)
                                                            .operation("exp(a[i])")
                                                            .build()
                                                            .generateCode()
                                            )
                                            .build()
                            )
                            .buildCode(), List.of("exp_kernel"));
            kernel.launchBlocking(
                    "exp_kernel",
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

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input) {
        Optional<ITensor> expOpt = ctx.getSavedTensor("exp");
        if (expOpt.isPresent()) {
            return expOpt.get();
        } else {
            ITensor result = exp(input);
            ctx.saveForBackward("exp", result);
            return result;
        }
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input) {
        return new LazyTensor(backend, input.getShape(), input.getDataType());
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient input) {
        if (input.requiresGradients()) {
            ITensor exp = ctx.getSavedTensorOrPopulateWith("exp", () -> exp(input.getValue()));
            input.accumulateGradient(upstreamGradient.multiply(exp));
        }
    }
}
