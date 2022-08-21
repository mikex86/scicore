package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.backend.impl.jvm.JvmDerivedTensor;
import me.mikex86.scicore.op.IBinaryOperation;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

public class JvmDividedOp implements IBinaryOperation {

    @NotNull
    private final ISciCoreBackend backend;

    public JvmDividedOp(@NotNull ISciCoreBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull ITensor a, @NotNull ITensor b) {
        long[] shapeA = a.getShape();
        long[] stridesA = a.getStrides();
        long[] shapeB = b.getShape();
        long[] stridesB = b.getStrides();
        long[] outputShape = ShapeUtils.broadcastShapes(shapeA, shapeB);
        long[] stridesOut = ShapeUtils.makeStrides(outputShape);

        DataType ownDataType = a.getDataType();
        DataType otherDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(ownDataType, otherDataType);
        ITensor result = backend.createTensor(resultDataType, outputShape);

        long[] outputIndex = new long[outputShape.length];
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
        } while (ShapeUtils.incrementIndex(outputShape, outputIndex));
        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull ITensor a, @NotNull ITensor b) {
        Validator.assertTrue(a.getDataType().isNumeric(), "A is not numeric");
        Validator.assertTrue(b.getDataType().isNumeric(), "B is not numeric");
        long[] shapeA = a.getShape();
        long[] shapeB = b.getShape();
        long[] outputShape = ShapeUtils.broadcastShapes(shapeA, shapeB);
        DataType ownDataType = a.getDataType();
        DataType otherDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(ownDataType, otherDataType);
        return new JvmDerivedTensor(backend, outputShape, resultDataType, () -> perform(a, b));
    }

}
