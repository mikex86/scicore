package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.memory.MemoryHandleGuard;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.backend.impl.genericcpu.jni.MinusJNI;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.graph.op.IInplaceOperation;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.utils.GradientUtil;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

public class GenCPUMinusInplaceOp implements IDifferentiableBinaryOperation, IInplaceOperation {

    @NotNull
    private final GenCPUBackend backend;

    public GenCPUMinusInplaceOp(@NotNull GenCPUBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shapeA = a.getShape();
        long[] shapeB = b.getShape();
        long[] finalShape = ShapeUtils.broadcastShapes(shapeA, shapeB);
        if (!ShapeUtils.equals(shapeA, finalShape)) {
            throw new IllegalArgumentException("Cannot perform inplace operation on tensor with shape " + ShapeUtils.toString(shapeA) + " and shape " + ShapeUtils.toString(finalShape));
        }

        long[] stridesA = a.getStrides();
        long[] stridesB = b.getStrides();

        DataType dataTypeA = a.getDataType();
        DataType dataTypeB = b.getDataType();
        DataType resultDataType = DataType.getLarger(dataTypeA, dataTypeB);
        if (resultDataType != dataTypeA) {
            throw new IllegalArgumentException("Cannot perform inplace operation on tensor with data type " + dataTypeA + " and data type " + resultDataType);
        }

        DirectMemoryHandle aMemoryHandle = a.getContentsAsDirectMemory();
        DirectMemoryHandle bMemoryHandle = b.getContentsAsDirectMemory();

        MinusJNI.minus(
                aMemoryHandle.getNativePtr(), shapeA, stridesA, a.getDataType(),
                bMemoryHandle.getNativePtr(), shapeB, stridesB, b.getDataType(),
                aMemoryHandle.getNativePtr(), finalShape, a.getStrides(), a.getDataType()
        );

        return a;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shapeA = a.getShape();
        long[] shapeB = b.getShape();
        long[] finalShape = ShapeUtils.broadcastShapes(shapeA, shapeB);
        if (!ShapeUtils.equals(shapeA, finalShape)) {
            throw new IllegalArgumentException("Cannot perform inplace operation on tensor with shape " + ShapeUtils.toString(shapeA) + " and shape " + ShapeUtils.toString(finalShape));
        }
        DataType dataTypeA = a.getDataType();
        DataType dataTypeB = b.getDataType();
        DataType resultDataType = DataType.getLarger(dataTypeA, dataTypeB);
        if (resultDataType != dataTypeA) {
            throw new IllegalArgumentException("Cannot perform inplace operation on tensor with data type " + dataTypeA + " and data type " + resultDataType);
        }
        return a;
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient a, @NotNull IGraph.ITensorNodeWithGradient b) {
        // Note that the upstream gradient dL/dz where z is the output of the current node
        // is with respect to all parameters a(p11,p12,...p1n) and b(p21,p22,...p2n) where a and b are the
        // inputs to the current node.
        // When computing the gradient for a, we need to sum over all the other parameters in b, and vice versa.
        if (a.requiresGradients()) {
            ITensor aValue = a.getValue();
            ITensor gradients = backend.createTensor(upstreamGradient.getDataType(), aValue.getShape());
            gradients.fill(1);
            gradients = gradients.multiply(upstreamGradient);
            gradients = GradientUtil.sumGradientsOnBroadcastDims(gradients, aValue.getShape());
            a.accumulateGradient(gradients);
        }
        if (b.requiresGradients()) {
            ITensor bValue = b.getValue();
            ITensor gradients = backend.createTensor(upstreamGradient.getDataType(), b.getValue().getShape());
            gradients.fill(-1);
            gradients = gradients.multiply(upstreamGradient);
            gradients = GradientUtil.sumGradientsOnBroadcastDims(gradients, bValue.getShape());
            b.accumulateGradient(gradients);
        }
    }
}
