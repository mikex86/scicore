package me.mikex86.scicore.backend.impl.cuda.op;

import jcuda.Pointer;
import me.mikex86.scicore.backend.impl.cuda.CudaBackend;
import me.mikex86.scicore.backend.impl.cuda.CudaTensor;
import me.mikex86.scicore.backend.impl.cuda.codegen.BroadcastingElementWiseOperationKernelCodeGenerator;
import me.mikex86.scicore.backend.impl.cuda.codegen.KernelCodeGenerator;
import me.mikex86.scicore.backend.impl.cuda.kernel.CudaKernel;
import me.mikex86.scicore.backend.impl.cuda.kernel.CudaKernelLaunchConfig;
import me.mikex86.scicore.backend.impl.cuda.memory.CudaMemoryHandle;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.utils.GradientUtil;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public class CudaPlusOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final CudaBackend backend;

    public CudaPlusOp(@NotNull CudaBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shapeA = a.getShape();
        long[] shapeB = b.getShape();

        long[] stridesA = a.getStrides();
        long[] stridesB = b.getStrides();

        long[] resultShape = ShapeUtils.broadcastShapes(shapeA, shapeB);

        DataType dataTypeA = a.getDataType();
        DataType dataTypeB = b.getDataType();
        DataType resultDataType = DataType.getLarger(dataTypeA, dataTypeB);

        long nElements = ShapeUtils.getNumElements(resultShape);

        CudaTensor result = this.backend.createTensor(resultDataType, resultShape);

        long[] resultStrides = result.getStrides();

        CudaMemoryHandle aDevicePtr = backend.getCudaMemoryManager().ensureOnDevice(a);
        CudaMemoryHandle bDevicePtr = backend.getCudaMemoryManager().ensureOnDevice(b);
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
                                        .functionName("plus")
                                        .parameter(dataTypeA, 1, "a")
                                        .parameter(dataTypeB, 1, "b")
                                        .parameter(resultDataType, 1, "out")
                                        .body(
                                                BroadcastingElementWiseOperationKernelCodeGenerator
                                                        .builder()
                                                        .shapeA(shapeA)
                                                        .shapeB(shapeB)
                                                        .resultShape(resultShape)
                                                        .stridesA(stridesA)
                                                        .stridesB(stridesB)
                                                        .resultStrides(resultStrides)
                                                        .operator("+")
                                                        .build()
                                                        .generateCode()
                                        )
                                        .build()
                        )
                        .buildCode(), List.of("plus"));
        kernel.launchOnStream(
                backend.getStream(),
                "plus",
                CudaKernelLaunchConfig.builder()
                        .blockDimX(threadsPerBlock)
                        .gridDimX(nBlocks)
                        .parameters(
                                Pointer.to(
                                        Pointer.to(aDevicePtr.getDevicePointer()),
                                        Pointer.to(bDevicePtr.getDevicePointer()),
                                        Pointer.to(result.getDataContainer().getDeviceMemoryHandle().getDevicePointer())
                                )
                        )
                        .build()
        );

        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shapeA = a.getShape();
        long[] shapeB = b.getShape();
        long[] finalShape = ShapeUtils.broadcastShapes(shapeA, shapeB);

        DataType dataTypeA = a.getDataType();
        DataType dataTypeB = b.getDataType();

        DataType resultDataType = DataType.getLarger(dataTypeA, dataTypeB);

        return new LazyTensor(backend, finalShape, resultDataType);
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient a, @NotNull IGraph.ITensorNodeWithGradient b) {
        if (a.requiresGradients()) {
            ITensor aValue = a.getValue();
            ITensor gradients = GradientUtil.sumGradientsOnBroadcastDims(upstreamGradient, aValue.getShape());
            a.accumulateGradient(gradients);
        }

        if (b.requiresGradients()) {
            ITensor bValue = b.getValue();
            ITensor gradients = GradientUtil.sumGradientsOnBroadcastDims(upstreamGradient, bValue.getShape());
            b.accumulateGradient(gradients);
        }
    }
}
