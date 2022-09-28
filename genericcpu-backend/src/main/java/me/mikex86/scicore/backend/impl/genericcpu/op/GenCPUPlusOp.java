package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.backend.impl.genericcpu.GenCPUTensor;
import me.mikex86.scicore.backend.impl.genericcpu.jni.PlusJNI;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.op.Graph;
import me.mikex86.scicore.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

public class GenCPUPlusOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final GenCPUBackend backend;

    public GenCPUPlusOp(@NotNull GenCPUBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shapeA = a.getShape();
        long[] shapeB = b.getShape();

        long aNumElements = ShapeUtils.getNumElements(shapeA);
        long bNumElements = ShapeUtils.getNumElements(shapeB);

        long[] finalShape = ShapeUtils.broadcastShapes(shapeA, shapeB);
        DataType aDataType = a.getDataType();
        DataType bDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(aDataType, bDataType);

        long nResultElements = ShapeUtils.getNumElements(finalShape);
        GenCPUTensor resultTensor = new GenCPUTensor(this.backend, resultDataType, finalShape);

        DirectMemoryHandle aMemory = backend.getMemoryManager().ensureDirect(a);
        DirectMemoryHandle bMemory = backend.getMemoryManager().ensureDirect(b);
        DirectMemoryHandle resultMemory = resultTensor.getDataContainer().getMemoryHandle();

        PlusJNI.plus(
                aMemory.getNativePtr(),
                aDataType,
                aNumElements,
                bMemory.getNativePtr(),
                bDataType,
                bNumElements,
                resultMemory.getNativePtr(),
                nResultElements,
                resultDataType
        );

        return resultTensor;
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
    public void computeGradients(Graph.@NotNull IOperationContext ctx, @NotNull ITensor upstreamGradient, IGraph.@NotNull ITensorNodeWithGradient a, IGraph.@NotNull ITensorNodeWithGradient b) {
        // Note that the upstream gradient dL/dz where z is the output of the current node
        // is with respect to all parameters a(p11,p12,...p1n) and b(p21,p22,...p2n) where a and b are the
        // inputs to the current node.
        // When computing the gradient for a, we need to sum over all the other parameters in b, and vice versa.

        if (a.requiresGradients()) {
            ITensor aValue = a.getValue();

            long[] shapeA = aValue.getShape();

            ITensor gradients = backend.createTensor(upstreamGradient.getDataType(), shapeA);
            gradients.fill(1);
            gradients = gradients.multiply(upstreamGradient);

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

            ITensor gradients = backend.createTensor(upstreamGradient.getDataType(), shapeB);
            gradients.fill(1);
            gradients = gradients.multiply(upstreamGradient);

            long[] gradientsShape = gradients.getShape();

            if (ShapeUtils.compareBroadcastRank(gradientsShape, shapeB) > 0) {
                int nCommonDimensions = ShapeUtils.getNumNotCommonDimensions(gradientsShape, shapeB);
                for (int i = 0; i < nCommonDimensions; i++) {
                    gradients = gradients.reduceSum(0);
                }
            }

            b.accumulateGradient(gradients);
        }
    }
}
