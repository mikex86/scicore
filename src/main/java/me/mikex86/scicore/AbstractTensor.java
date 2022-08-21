package me.mikex86.scicore;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.op.IGraphRecorder;
import me.mikex86.scicore.op.OperationType;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;

public abstract class AbstractTensor implements ITensor {

    @Override
    @NotNull
    public ITensor getView(long @NotNull ... indices) {
        long[] shape = getShape();
        validateIndices(indices);
        long[] strides = ShapeUtils.makeStrides(shape);

        long[] sliceShape = Arrays.copyOfRange(shape, indices.length, shape.length);
        long[] sliceStrides = Arrays.copyOfRange(strides, indices.length, strides.length);

        long offset = ShapeUtils.getFlatIndex(indices, strides);
        return new View(this, sliceShape, offset, sliceStrides);
    }

    @Override
    @NotNull
    public ITensor matmul(@NotNull ITensor other) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.MATMUL, this, other);
    }

    @Override
    @NotNull
    public ITensor divided(@NotNull ITensor other) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.DIVIDED, this, other);
    }

    @Override
    @NotNull
    public ITensor plus(@NotNull ITensor other) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.PLUS, this, other);
    }

    @Override
    @NotNull
    public ITensor exp() {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.EXP, this);
    }

    @Override
    @NotNull
    public ITensor softmax(int dimension) {
        ITensor exponentiated = exp();
        ITensor sum = exponentiated.reduceSum(dimension, true);
        return exponentiated.divided(sum);
    }

    @Override
    @NotNull
    public ITensor reduceSum(int dimension) {
        return reduceSum(dimension, false);
    }

    @Override
    @NotNull
    public ITensor reduceSum(int dimension, boolean keepDimensions) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor dimensionScalar;
        {
            dimensionScalar = backend.createTensor(DataType.INT32, new long[]{1});
            dimensionScalar.setIntFlat(dimension, 0);
        }
        ITensor keepDimensionsScalar;
        {
            keepDimensionsScalar = backend.createTensor(DataType.BOOLEAN, new long[]{1});
            keepDimensionsScalar.setBooleanFlat(keepDimensions, 0);
        }
        return operationRecorder.recordOperation(OperationType.REDUCE_SUM, this, dimensionScalar, keepDimensionsScalar);
    }

    @Override
    @NotNull
    public ITensor transpose() {
        // TODO: OPTIMIZE TRANSPOSE AS TRANSPOSED VIEW
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.TRANSPOSE, this);
    }

    protected void validateDataType(@NotNull DataType requestedDataType) {
        DataType ownDataType = getDataType();
        if (requestedDataType != ownDataType) {
            throw new IllegalArgumentException("Requested data type " + requestedDataType + " does not match data type of viewed tensor " + ownDataType);
        }
    }

    protected void validateIndices(long @NotNull [] indices) {
        long[] shape = getShape();
        if (indices.length > shape.length) {
            throw new IllegalArgumentException("Indices length is greater than tensor shape length");
        }
        for (int i = 0; i < indices.length; i++) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException("Index " + indices[i] + " is out of bounds for dimension " + i + " with shape " + shape[i]);
            }
        }
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (!(obj instanceof ITensor other)) {
            return false;
        }
        long[] shape = getShape();
        if (!ShapeUtils.equals(shape, other.getShape())) {
            return false;
        }
        long nElements = ShapeUtils.getNumElements(shape);
        boolean oneIsFloatingPoint = getDataType().isFloatingPoint() || other.getDataType().isFloatingPoint();
        if (oneIsFloatingPoint) {
            for (long i = 0; i < nElements; i++) {
                double a = getAsDoubleFlat(i);
                double b = other.getAsDoubleFlat(i);
                if (Math.abs(a - b) > EPSILON) {
                    return false;
                }
            }
        } else {
            for (long i = 0; i < nElements; i++) {
                long a = getAsLongFlat(i);
                long b = getAsLongFlat(i);
                if (a != b) {
                    return false;
                }
            }
        }
        return true;
    }
}
