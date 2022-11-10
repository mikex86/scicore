package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.op.IBinaryOperation;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

public class JvmCompareElementsOp implements IBinaryOperation {

    @NotNull
    private final JvmBackend backend;

    public JvmCompareElementsOp(@NotNull JvmBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shapeA = a.getShape();
        long[] shapeB = b.getShape();

        long[] stridesA = a.getStrides();
        long[] stridesB = b.getStrides();

        Validator.assertTrue(ShapeUtils.equals(shapeA, shapeB), "Shapes must be equal");

        ITensor result = backend.createTensor(DataType.BOOLEAN, shapeA);
        long[] resultStrides = result.getStrides();

        long[] indexA = new long[shapeA.length];
        long[] indexB = new long[shapeB.length];
        long[] outputIndex = new long[shapeA.length];

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

            if (a.getDataType().isNumeric() && b.getDataType().isNumeric()) {
                if (a.getDataType().isFloatingPoint() && b.getDataType().isFloatingPoint()) {
                    double valueA = a.getAsDouble(indexAFlat);
                    double valueB = b.getAsDouble(indexBFlat);
                    result.setBoolean(valueA == valueB, outputIndexFlat);
                } else if (a.getDataType().isFloatingPoint()) {
                    double valueA = a.getAsDouble(indexAFlat);
                    long valueB = b.getAsLong(indexBFlat);
                    result.setBoolean(valueA == valueB, outputIndexFlat);
                } else if (b.getDataType().isFloatingPoint()) {
                    long valueA = a.getAsLong(indexAFlat);
                    double valueB = b.getAsDouble(indexBFlat);
                    result.setBoolean(valueA == valueB, outputIndexFlat);
                } else {
                    long valueA = a.getAsLong(indexAFlat);
                    long valueB = b.getAsLong(indexBFlat);
                    result.setBoolean(valueA == valueB, outputIndexFlat);
                }
            } else if (a.getDataType() == DataType.BOOLEAN && b.getDataType() == DataType.BOOLEAN) {
                boolean valueA = a.getBoolean(indexAFlat);
                boolean valueB = b.getBoolean(indexBFlat);
                result.setBoolean(valueA == valueB, outputIndexFlat);
            } else {
                throw new IllegalArgumentException("Unsupported data types: " + a.getDataType() + " and " + b.getDataType());
            }
        } while (ShapeUtils.incrementIndex(outputIndex, shapeA));
        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shapeA = a.getShape();
        long[] shapeB = b.getShape();
        Validator.assertTrue(ShapeUtils.equals(shapeA, shapeB), "Shapes must be equal");
        return new LazyTensor(backend, shapeA, DataType.BOOLEAN);
    }
}
