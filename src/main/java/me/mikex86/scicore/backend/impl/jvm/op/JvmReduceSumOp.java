package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.op.IBiParametricOperation;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;

public class JvmReduceSumOp implements IBiParametricOperation<Integer, Boolean> {

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
    public ITensor perform(@NotNull ITensor tensor, Integer dimension, Boolean keepDimensions) {
        // TODO: OPTIMIZE
        DataType dataType = tensor.getDataType();
        long[] shape = tensor.getShape();
        if (dimension == -1) {
            long[] outputShape;
            if (keepDimensions) {
                outputShape = new long[shape.length];
            } else {
                outputShape = new long[1];
            }
            Arrays.fill(outputShape, 1);

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
            // increment index, but only for dimensions that are not being summed along
            {
                boolean hasNext = false;
                for (int dim = 0; dim < completeIndex.length; dim++) {
                    if (dim == dimension) {
                        continue;
                    }
                    if (completeIndex[dim] < shape[dim] - 1) {
                        completeIndex[dim]++;
                        if (dim > dimension) {
                            reducedIndex[dim - 1] = completeIndex[dim];
                        } else {
                            reducedIndex[dim] = completeIndex[dim];
                        }
                        hasNext = true;
                        break;
                    }
                    completeIndex[dim] = 0;
                }
                if (!hasNext) {
                    break;
                }
            }
        }
        return result;
    }

    @NotNull
    @Override
    public ITensor performLazily(@NotNull ITensor tensor, Integer dimension, Boolean keepDimensions) {
        DataType dataType = tensor.getDataType();
        long[] shape = tensor.getShape();
        long[] outputShape;
        if (dimension == -1) {
            if (keepDimensions) {
                outputShape = new long[shape.length];
            } else {
                outputShape = new long[1];
            }
            Arrays.fill(outputShape, 1);
        } else {
            outputShape = new long[shape.length - (keepDimensions ? 0 : 1)];
            reduceShape(shape, outputShape, dimension, keepDimensions);
        }
        return new LazyTensor(backend, outputShape, dataType, () -> perform(tensor, dimension, keepDimensions));
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

    @Override
    public @NotNull Class<Integer> getFirstType() {
        return Integer.class;
    }

    @Override
    public @NotNull Class<Boolean> getSecondType() {
        return Boolean.class;
    }
}
