package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.backend.impl.genericcpu.jni.MultiplyJNI;
import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.backend.impl.jvm.JvmTensor;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.op.Graph;
import me.mikex86.scicore.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

public class GenCPUMultiplyOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final GenCPUBackend backend;

    public GenCPUMultiplyOp(@NotNull GenCPUBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shapeA = a.getShape();
        long[] shapeB = b.getShape();
        long[] finalShape = ShapeUtils.broadcastShapes(shapeA, shapeB);

        long[] stridesA = a.getStrides();
        long[] stridesB = b.getStrides();

        DataType ownDataType = a.getDataType();
        DataType otherDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(ownDataType, otherDataType);
        ITensor result = backend.createTensor(resultDataType, finalShape);
        long[] resultStrides = result.getStrides();

        DirectMemoryHandle aMemoryHandle = a.getContentsAsDirectMemory();
        DirectMemoryHandle bMemoryHandle = b.getContentsAsDirectMemory();
        DirectMemoryHandle resultMemoryHandle = result.getContentsAsDirectMemory();

        MultiplyJNI.multiply(
                aMemoryHandle.getNativePtr(), shapeA, stridesA, a.getDataType(),
                bMemoryHandle.getNativePtr(), shapeB, stridesB, b.getDataType(),
                resultMemoryHandle.getNativePtr(), finalShape, resultStrides, result.getDataType());

        return result;
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
