package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.utils.GradientUtil;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

public class JvmMultiplyOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final JvmBackend backend;

    public JvmMultiplyOp(@NotNull JvmBackend backend) {
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
                double aDivB = aV * bV;
                result.setByDoubleFlat(aDivB, outputIndexFlat);
            } else {
                long aV = a.getAsLongFlat(indexAFlat);
                long bV = b.getAsLongFlat(indexBFlat);
                long aDivB = aV * bV;
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
        return new LazyTensor(backend, finalShape, resultDataType);
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient a, @NotNull IGraph.ITensorNodeWithGradient b) {
        if (a.requiresGradients()) {
            ITensor gradients = upstreamGradient.multiply(b.getValue());
            gradients = GradientUtil.sumGradientsOnBroadcastDims(gradients, a.getValue().getShape());
            a.accumulateGradient(gradients);
        }
        if (b.requiresGradients()) {
            ITensor gradients = upstreamGradient.multiply(a.getValue());
            gradients = GradientUtil.sumGradientsOnBroadcastDims(gradients, b.getValue().getShape());
            b.accumulateGradient(gradients);
        }
    }
}
