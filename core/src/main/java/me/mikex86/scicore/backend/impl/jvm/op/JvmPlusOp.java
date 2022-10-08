package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.backend.impl.jvm.JvmTensor;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.op.Graph;
import me.mikex86.scicore.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

public class JvmPlusOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final JvmBackend backend;

    public JvmPlusOp(@NotNull JvmBackend backend) {
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

        long[] outputIndex = new long[finalShape.length];
        long[] indexA = new long[shapeA.length];
        long[] indexB = new long[shapeB.length];

        do {
            // copy common dimensions into indexA and indexB
            for (int i = 0; i < indexA.length; i++) {
                indexA[indexA.length - 1 - i] = outputIndex[outputIndex.length - 1 - i];
            }
            for (int i = 0; i < indexB.length; i++) {
                indexB[indexB.length - 1 - i] = outputIndex[outputIndex.length - 1 - i];
            }
            // constrain indices
            ShapeUtils.constrainIndex(indexA, shapeA);
            ShapeUtils.constrainIndex(indexB, shapeB);

            long outputIndexFlat = ShapeUtils.getFlatIndex(outputIndex, resultStrides);
            long indexAFlat = ShapeUtils.getFlatIndex(indexA, stridesA);
            long indexBFlat = ShapeUtils.getFlatIndex(indexB, stridesB);

            if (resultDataType.isFloatingPoint()) {
                double aV = a.getAsDoubleFlat(indexAFlat);
                double bV = b.getAsDoubleFlat(indexBFlat);
                double aDivB = aV + bV;
                result.setByDoubleFlat(aDivB, outputIndexFlat);
            } else {
                long aV = a.getAsLongFlat(indexAFlat);
                long bV = b.getAsLongFlat(indexBFlat);
                long aDivB = aV + bV;
                result.setByLongFlat(aDivB, outputIndexFlat);
            }
        } while (ShapeUtils.incrementIndex(outputIndex, finalShape));
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
