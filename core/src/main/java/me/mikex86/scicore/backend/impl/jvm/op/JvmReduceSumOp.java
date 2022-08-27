package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.op.Graph;
import me.mikex86.scicore.op.IDifferentiableBiParametricOperation;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;

public class JvmReduceSumOp implements IDifferentiableBiParametricOperation<Integer, Boolean> {

    @NotNull
    private final ISciCoreBackend backend;

    public JvmReduceSumOp(@NotNull ISciCoreBackend backend) {
        this.backend = backend;
    }

    @NotNull
    public ISciCoreBackend getBackend() {
        return backend;
    }

    @NotNull
    @Override
    public ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor tensor, @Nullable Integer dimension, @Nullable Boolean keepDimensions) {
        Validator.validateNotNull(dimension, "Dimension must not be null");
        Validator.validateNotNull(keepDimensions, "KeepDimensions must not be null");

        DataType dataType = tensor.getDataType();
        long[] shape = tensor.getShape();
        if (dimension == -1) {
            long[] outputShape = getOutputShape(shape, dimension, keepDimensions);

            ITensor result = backend.createTensor(dataType, outputShape);
            long numElements = ShapeUtils.getNumElements(shape);
            if (dataType.isFloatingPoint()) {
                double sum = 0;
                for (long i = 0; i < numElements; i++) {
                    sum += tensor.getAsDoubleFlat(i);
                }
                result.setByDoubleFlat(sum, 0);
            } else {
                long sum = 0;
                for (long i = 0; i < numElements; i++) {
                    sum += tensor.getAsLongFlat(i);
                }
                result.setByLongFlat(sum, 0);
            }
            return result;
        }
        if (dimension < 0 || dimension >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds: " + dimension);
        }
        long[] reducedShape = new long[shape.length - (keepDimensions ? 0 : 1)];
        reduceShape(shape, reducedShape, dimension, keepDimensions);

        ITensor result = backend.createTensor(dataType, reducedShape);

        long[] completeIndex = new long[shape.length];
        long[] reducedIndex = new long[reducedShape.length];

        while (true) {
            if (dataType.isFloatingPoint()) {
                double sum = 0;
                for (long i = 0; i < shape[dimension]; i++) {
                    completeIndex[dimension] = i;
                    sum += tensor.getAsDouble(completeIndex);
                }

                result.setByDouble(sum, reducedIndex);
            } else {
                long sum = 0;

                for (long i = 0; i < shape[dimension]; i++) {
                    completeIndex[dimension] = i;
                    sum += tensor.getAsLong(completeIndex);
                }

                result.setLong(sum, reducedIndex);
            }
            if (!ShapeUtils.incrementIndex(reducedShape, reducedIndex)) {
                break;
            }
            if (!ShapeUtils.incrementIndex(shape, completeIndex)) {
                break;
            }
        }
        return result;
    }

    @NotNull
    @Override
    public ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor tensor, @Nullable Integer dimension, @Nullable Boolean keepDimensions) {
        Validator.validateNotNull(dimension, "Dimension must not be null");
        Validator.validateNotNull(keepDimensions, "KeepDimensions must not be null");
        DataType dataType = tensor.getDataType();
        long[] shape = tensor.getShape();
        long[] outputShape = getOutputShape(shape, dimension, keepDimensions);
        return new LazyTensor(backend, outputShape, dataType, () -> perform(ctx, tensor, dimension, keepDimensions));
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.IDifferentiableNode node, @Nullable Integer dimension, @Nullable Boolean keepDimensions) {
        Validator.validateNotNull(dimension, "Dimension must not be null");
        Validator.validateNotNull(keepDimensions, "KeepDimensions must not be null");

        if (node.requiresGradients()) {
            ITensor gradients = backend.createTensor(upstreamGradient.getDataType(), node.getValue().getShape());
            gradients.fill(1);
            gradients = gradients.multiply(upstreamGradient);

            node.accumulateGradient(gradients);
        }
    }

    private static void reduceShape(long[] shape, long[] outputShape, Integer dimension, Boolean keepDimensions) {
        for (int i = 0; i < shape.length; i++) {
            long dimSize = shape[i];
            if (keepDimensions) {
                if (i == dimension) {
                    dimSize = 1;
                }
                outputShape[i] = dimSize;
            } else {
                if (i < dimension) {
                    outputShape[i] = dimSize;
                } else if (i > dimension) {
                    outputShape[i - 1] = dimSize;
                }
            }
        }
    }


    private static long[] getOutputShape(long[] srcShape, int dimension, boolean keepDimensions) {
        long[] outputShape;
        if (dimension == -1) {
            if (keepDimensions) {
                outputShape = new long[srcShape.length];
            } else {
                outputShape = new long[0];
            }
            Arrays.fill(outputShape, 1);
        } else {
            if (dimension < 0 || dimension >= srcShape.length) {
                throw new IllegalArgumentException("Dimension out of bounds: " + dimension);
            }
            outputShape = new long[srcShape.length - (keepDimensions ? 0 : 1)];
            reduceShape(srcShape, outputShape, dimension, keepDimensions);
        }
        return outputShape;
    }

    @Override
    public @NotNull Class<Integer> getFirstType() {
        return Integer.class;
    }

    @Override
    public @NotNull Class<Boolean> getSecondType() {
        return Boolean.class;
    }
}
