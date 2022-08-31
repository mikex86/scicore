package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.op.Graph;
import me.mikex86.scicore.op.IDifferentiableSingleParametricOperation;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.Locale;

public class JvmArgmaxOp implements IDifferentiableSingleParametricOperation<Integer> {

    @NotNull
    private final JvmBackend backend;

    public JvmArgmaxOp(@NotNull JvmBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor tensor, @Nullable Integer dimension) {
        Validator.assertTrue(dimension != null, "dimensionScalar must not be null");

        DataType inputDataType = tensor.getDataType();
        long[] shape = tensor.getShape();
        if (dimension == -1) {
            long[] outputShape = ShapeUtils.getReducedShape(shape, dimension, false);

            ITensor result = backend.createTensor(DataType.INT64, outputShape);
            long numElements = ShapeUtils.getNumElements(shape);
            if (inputDataType.isFloatingPoint()) {
                double max = Double.MIN_VALUE;
                long maxIndex = 0;
                for (long i = 0; i < numElements; i++) {
                    double value = tensor.getAsDoubleFlat(i);
                    if (value > max) {
                        max = value;
                        maxIndex = i;
                    }
                }
                result.setByDoubleFlat(maxIndex, 0);
            } else {
                long max = Long.MIN_VALUE;
                long maxIndex = 0;
                for (long i = 0; i < numElements; i++) {
                    long value = tensor.getAsLongFlat(i);
                    if (value > max) {
                        max = value;
                        maxIndex = i;
                    }
                }
                result.setByLongFlat(maxIndex, 0);
            }
            return result;
        }
        if (dimension < 0 || dimension >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds: " + dimension);
        }
        long[] reducedShape = new long[shape.length - 1];
        ShapeUtils.reduceShape(shape, reducedShape, dimension, false);

        ITensor result = backend.createTensor(DataType.INT64, reducedShape);

        long[] completeIndex = new long[shape.length];
        long[] reducedIndex = new long[reducedShape.length];

        while (true) {
            if (inputDataType.isFloatingPoint()) {
                double max = Double.MIN_VALUE;
                long maxIndex = 0;
                for (long i = 0; i < shape[dimension]; i++) {
                    completeIndex[dimension] = i;
                    double value = tensor.getAsDouble(completeIndex);
                    if (value > max) {
                        max = value;
                        maxIndex = i;
                    }
                }
                result.setLong(maxIndex, reducedIndex);
            } else {
                long max = Long.MIN_VALUE;
                long maxIndex = 0;
                for (long i = 0; i < shape[dimension]; i++) {
                    completeIndex[dimension] = i;
                    long value = tensor.getAsLong(completeIndex);
                    if (value > max) {
                        max = value;
                        maxIndex = i;
                    }
                }
                result.setLong(maxIndex, reducedIndex);
            }
            if (!ShapeUtils.incrementIndex(reducedIndex, reducedShape)) {
                break;
            }
            if (!ShapeUtils.incrementIndex(completeIndex, shape)) {
                break;
            }
        }
        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input, @Nullable Integer dimension) {
        Validator.assertTrue(dimension != null, "dimensionScalar must not be null");
        long[] reducedShape = ShapeUtils.getReducedShape(input.getShape(), dimension, false);
        return new LazyTensor(backend, reducedShape, DataType.INT64, () -> perform(ctx, input, dimension));
    }


    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.IDifferentiableNode node, @Nullable Integer integer) {
        // TODO: IMPLEMENT
        throw new UnsupportedOperationException("TODO: implement");
    }

    @Override
    public @NotNull Class<Integer> getType() {
        return Integer.class;
    }
}