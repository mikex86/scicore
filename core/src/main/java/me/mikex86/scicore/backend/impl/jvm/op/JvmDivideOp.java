package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.op.Graph;
import me.mikex86.scicore.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

public class JvmDivideOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final ISciCoreBackend backend;

    public JvmDivideOp(@NotNull ISciCoreBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shapeA = a.getShape();
        long[] shapeB = b.getShape();
        long[] finalShape = ShapeUtils.broadcastShapes(shapeA, shapeB);

        long[] stridesA = a.getStrides();
        long[] stridesB = b.getStrides();
        long[] stridesOut = ShapeUtils.makeStrides(finalShape);

        DataType ownDataType = a.getDataType();
        DataType otherDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(ownDataType, otherDataType);
        ITensor result = backend.createTensor(resultDataType, finalShape);

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

            long outputIndexFlat = ShapeUtils.getFlatIndex(outputIndex, stridesOut);
            long indexAFlat = ShapeUtils.getFlatIndex(indexA, stridesA);
            long indexBFlat = ShapeUtils.getFlatIndex(indexB, stridesB);

            if (resultDataType.isFloatingPoint()) {
                double aV = a.getAsDoubleFlat(indexAFlat);
                double bV = b.getAsDoubleFlat(indexBFlat);
                double aDivB = aV / bV;
                result.setByDoubleFlat(aDivB, outputIndexFlat);
            } else {
                long aV = a.getAsLongFlat(indexAFlat);
                long bV = b.getAsLongFlat(indexBFlat);
                long aDivB = aV / bV;
                result.setByLongFlat(aDivB, outputIndexFlat);
            }
        } while (ShapeUtils.incrementIndex(outputIndex, finalShape));
        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        Validator.assertTrue(a.getDataType().isNumeric(), "A is not numeric");
        Validator.assertTrue(b.getDataType().isNumeric(), "B is not numeric");
        long[] shapeA = a.getShape();
        long[] shapeB = b.getShape();
        long[] finalShape = ShapeUtils.broadcastShapes(shapeA, shapeB);

        DataType ownDataType = a.getDataType();
        DataType otherDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(ownDataType, otherDataType);
        return new LazyTensor(backend, finalShape, resultDataType, () -> perform(ctx, a, b));
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradients, @NotNull IGraph.ITensorNodeWithGradient a, @NotNull IGraph.ITensorNodeWithGradient b) {
        // R = A / B
        // dL/dR = upstream gradients
        ITensor A = a.getValue();
        ITensor B = b.getValue();
        if (a.requiresGradients()) {
            // dL/dA = dL/dR * dR/dA
            // dR/dA = 1/dB
            // dL/dA = dL/dR * 1/B
            ITensor dRdA = backend.createTensor(DataType.getLarger(A.getDataType(), B.getDataType()), B.getShape());
            dRdA.fill(1);
            dRdA = dRdA.divide(B); // TODO: optimize this with an leftDivide operation so that you can do B.leftDiv(1) to express 1/B
            a.accumulateGradient(upstreamGradients.multiply(dRdA));
        }
        if (b.requiresGradients()) {
            // dL/dB = dL/dR * dR/dB
            // R = A * B^-1
            // dR/dB = -A * (B^-2)
            // dR/dB = -A * (1/B^2)
            // dL/dB = dL/dR * -A * (1/B^2)
            ITensor dRdB = backend.createTensor(B.getDataType(), B.getShape());
            dRdB.fill(1);
            dRdB = dRdB.divide(B.pow(2));
            dRdB = A.multiply(-1).multiply(dRdB);
            b.accumulateGradient(upstreamGradients.multiply(dRdB));
        }
    }
}
