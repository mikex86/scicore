package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.utils.GradientUtil;
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
        return new LazyTensor(backend, finalShape, resultDataType);
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
            try (ITensor dRdA = B.leftDivide(1f)) {
                try (ITensor dLdATmp = upstreamGradients.multiply(dRdA)) {
                    ITensor dLdAFinal = GradientUtil.sumGradientsOnBroadcastDims(dLdATmp, A.getShape());
                    a.accumulateGradient(dLdAFinal);
                }
            }
        }
        if (b.requiresGradients()) {
            // dL/dB = dL/dR * dR/dB
            // R = A * B^-1
            // dR/dB = -A * (B^-2)
            // dL/dB = dL/dR * -A * (B^-2)
            try (ITensor bPowNeg2 = B.pow(-2f)) {
                try (ITensor negativeA = A.multiply(-1)) {
                    try (ITensor dRdB = negativeA.multiply(bPowNeg2)) {
                        try (ITensor dLdBTmp = upstreamGradients.multiply(dRdB)) {
                            ITensor dLdB = GradientUtil.sumGradientsOnBroadcastDims(dLdBTmp, B.getShape());
                            b.accumulateGradient(dLdB);
                        }
                    }
                }
            }
        }
    }
}
