package me.mikex86.scicore.backend.impl.cuda.op;

import jcuda.Pointer;
import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.backend.impl.cuda.CudaBackend;
import me.mikex86.scicore.backend.impl.cuda.CudaTensor;
import me.mikex86.scicore.backend.impl.cuda.kernel.CudaKernel;
import me.mikex86.scicore.backend.impl.cuda.kernel.CudaKernelLaunchConfig;
import me.mikex86.scicore.backend.impl.cuda.kernel.KernelNameUtility;
import me.mikex86.scicore.op.Graph;
import me.mikex86.scicore.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public class CudaMultiplyOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final CudaBackend backend;

    public CudaMultiplyOp(@NotNull CudaBackend backend) {
        this.backend = backend;
    }

    @NotNull
    private final CudaKernel multiplyKernel = CudaKernel.loadClassPath("kernels/cuda/multiply.ptx", KernelNameUtility.getAllTypePermutations("multiply", List.of(DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64, DataType.FLOAT32, DataType.FLOAT64)));

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shapeA = a.getShape();
        long[] shapeB = b.getShape();

        long aNumElements = ShapeUtils.getNumElements(shapeA);
        long bNumElements = ShapeUtils.getNumElements(shapeB);

        long[] finalShape = shapeA;
        boolean broadcast = false;
        if (!ShapeUtils.equals(shapeA, shapeB)) {
            finalShape = ShapeUtils.broadcastShapes(shapeA, shapeB);
            broadcast = true;
        }

        DataType dataTypeA = a.getDataType();
        DataType dataTypeB = b.getDataType();
        DataType resultDataType = DataType.getLarger(dataTypeA, dataTypeB);

        long nElements = ShapeUtils.getNumElements(finalShape);

        CudaTensor tensor = new CudaTensor(this.backend, resultDataType, finalShape);

        if (broadcast) {
            CudaTensor aTensor = a.getIfIsType(CudaTensor.class);
            CudaTensor bTensor = b.getIfIsType(CudaTensor.class);
            if (aTensor != null && bTensor != null) {
                int threadsPerBlock = 1024;
                int nBlocks = Math.toIntExact((nElements + threadsPerBlock - 1) / threadsPerBlock);
                this.multiplyKernel.launch(
                        KernelNameUtility.getTypePermutation("multiply", dataTypeA, dataTypeB),
                        CudaKernelLaunchConfig.builder()
                                .blockDimX(threadsPerBlock)
                                .gridDimX(nBlocks)
                                .parameters(
                                        Pointer.to(
                                                Pointer.to(aTensor.getDataContainer().getDeviceMemoryHandle().getPointer()),
                                                Pointer.to(new long[]{aNumElements}),
                                                Pointer.to(bTensor.getDataContainer().getDeviceMemoryHandle().getPointer()),
                                                Pointer.to(new long[]{bNumElements}),
                                                Pointer.to(tensor.getDataContainer().getDeviceMemoryHandle().getPointer()),
                                                Pointer.to(new long[]{nElements})
                                        )
                                )
                                .build()
                );
            }
        } else {
            for (long i = 0; i < nElements; i++) {
                if (resultDataType.isFloatingPoint()) {
                    double aV = a.getAsDoubleFlat(i);
                    double bV = b.isScalar() ? b.elementAsDouble() : b.getAsDoubleFlat(i);
                    double resultVal = aV * bV;
                    tensor.setByDoubleFlat(resultVal, i);
                } else {
                    long aV = a.getAsLongFlat(i);
                    long bV = b.isScalar() ? b.elementAsLong() : b.getAsLongFlat(i);
                    long resultVal = aV * bV;
                    tensor.setByLongFlat(resultVal, i);
                }
            }
        }
        return tensor;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shapeA = a.getShape();
        long[] finalShape = shapeA;
        long[] shapeB = b.getShape();
        if (!ShapeUtils.equals(shapeA, shapeB)) {
            finalShape = ShapeUtils.broadcastShapes(shapeA, shapeB);
        }

        DataType dataTypeA = a.getDataType();
        DataType dataTypeB = b.getDataType();

        DataType resultDataType = DataType.getLarger(dataTypeA, dataTypeB);

        return new LazyTensor(backend, finalShape, resultDataType, () -> perform(ctx, a, b));
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient a, @NotNull IGraph.ITensorNodeWithGradient b) {
        if (a.requiresGradients()) {
            ITensor aValue = a.getValue();

            long[] shapeA = aValue.getShape();

            ITensor gradients = upstreamGradient.multiply(b.getValue());

            long[] gradientShape = gradients.getShape();

            if (ShapeUtils.compareBroadcastRank(gradientShape, shapeA) > 0) {
                int nCommonDimensions = ShapeUtils.getNumNotCommonDimensions(shapeA, gradientShape);
                for (int i = 0; i < nCommonDimensions; i++) {
                    gradients = gradients.reduceSum(0);
                }
            }
            a.accumulateGradient(gradients);
        }
        if (b.requiresGradients()) {
            ITensor bValue = b.getValue();

            long[] shapeB = bValue.getShape();

            ITensor gradients = upstreamGradient.multiply(a.getValue());

            long[] gradientShape = gradients.getShape();

            if (ShapeUtils.compareBroadcastRank(gradientShape, shapeB) > 0) {
                int nCommonDimensions = ShapeUtils.getNumNotCommonDimensions(shapeB, gradientShape);
                for (int i = 0; i < nCommonDimensions; i++) {
                    gradients = gradients.reduceSum(0);
                }
            }

            b.accumulateGradient(gradients);
        }
    }
}
