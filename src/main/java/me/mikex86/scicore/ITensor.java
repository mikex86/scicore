package me.mikex86.scicore;

import me.mikex86.scicore.backend.ScalarImpl;
import me.mikex86.scicore.backend.SciCoreBackend;
import me.mikex86.scicore.backend.TensorImpl;
import me.mikex86.scicore.backend.impl.jvm.JvmTensorImpl;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.awt.*;
import java.util.Arrays;

public interface ITensor extends IValue {

    default void validateDataType(@NotNull DataType requestedDataType) {
        DataType ownDataType = getDataType();
        if (requestedDataType != ownDataType) {
            throw new IllegalArgumentException("Requested data type " + requestedDataType + " does not match data type of viewed tensor " + ownDataType);
        }
    }

    default void validateIndices(long @NotNull [] indices) {
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

    @NotNull DataType getDataType();

    long @NotNull [] getShape();

    @NotNull
    default ITensor getView(long @NotNull ... indices) {
        long[] shape = getShape();
        validateIndices(indices);
        long[] strides = ShapeUtils.makeStrides(shape);

        long[] sliceShape = Arrays.copyOfRange(shape, indices.length, shape.length);
        long[] sliceStrides = Arrays.copyOfRange(strides, indices.length, strides.length);

        long offset = ShapeUtils.getFlatIndex(indices, strides);
        return new View(this, sliceShape, offset, sliceStrides);
    }

    byte getByte(long @NotNull ... indices);

    short getShort(long @NotNull ... indices);

    int getInt(long @NotNull ... indices);

    long getLong(long @NotNull ... indices);

    float getFloat(long @NotNull ... indices);

    double getDouble(long @NotNull ... indices);

    void setByte(byte value, long @NotNull ... indices);

    void setShort(short value, long @NotNull ... indices);

    void setInt(int value, long @NotNull ... indices);

    void setLong(long value, long @NotNull ... indices);

    void setFloat(float value, long @NotNull ... indices);

    void setDouble(double value, long @NotNull ... indices);

    byte getByteFlat(long flatIndex);

    short getShortFlat(long flatIndex);

    int getIntFlat(long flatIndex);

    long getLongFlat(long flatIndex);

    float getFloatFlat(long flatIndex);

    double getDoubleFlat(long flatIndex);

    void setByteFlat(byte value, long flatIndex);

    void setShortFlat(short value, long flatIndex);

    void setIntFlat(int value, long flatIndex);

    void setLongFlat(long value, long flatIndex);

    void setFloatFlat(float value, long flatIndex);

    void setDoubleFlat(double value, long flatIndex);

    @NotNull ITensor copy();

    void setContents(@NotNull ITensor tensor);

    void setContents(long @NotNull [] dimension, @NotNull ITensor tensor, boolean useView);

    default long getNumberOfElements() {
        return ShapeUtils.getNumElements(getShape());
    }

    @NotNull
    default ITensor multiplied(@NotNull Scalar s) {
        // General multiply
        ITensor copy = copy();
        long nElements = copy.getNumberOfElements();
        DataType tensorDataType = copy.getDataType();
        DataType scalarDataType = s.getDataType();
        ScalarImpl scalarImpl = s.getScalarImpl();

        switch (tensorDataType) {
            case INT8 -> {
                switch (scalarDataType) {
                    case INT8 -> {
                        byte scalarValue = scalarImpl.getByte();
                        for (long i = 0; i < nElements; i++) {
                            copy.setByteFlat((byte) (copy.getByteFlat(i) * scalarValue), i);
                        }
                    }
                    case INT16 -> {
                        short scalarValue = scalarImpl.getShort();
                        for (long i = 0; i < nElements; i++) {
                            copy.setByteFlat((byte) (copy.getByteFlat(i) * scalarValue), i);
                        }
                    }
                    case INT32 -> {
                        int scalarValue = scalarImpl.getInt();
                        for (long i = 0; i < nElements; i++) {
                            copy.setByteFlat((byte) (copy.getByteFlat(i) * scalarValue), i);
                        }
                    }
                    case INT64 -> {
                        long scalarValue = scalarImpl.getLong();
                        for (long i = 0; i < nElements; i++) {
                            copy.setByteFlat((byte) (copy.getByteFlat(i) * scalarValue), i);
                        }
                    }
                    case FLOAT32 -> {
                        float scalarValue = scalarImpl.getFloat();
                        for (long i = 0; i < nElements; i++) {
                            copy.setByteFlat((byte) (copy.getByteFlat(i) * scalarValue), i);
                        }
                    }
                    case FLOAT64 -> {
                        double scalarValue = scalarImpl.getDouble();
                        for (long i = 0; i < nElements; i++) {
                            copy.setByteFlat((byte) (copy.getByteFlat(i) * scalarValue), i);
                        }
                    }
                }
            }
            case INT16 -> {
                switch (scalarDataType) {
                    case INT8 -> {
                        byte scalarValue = scalarImpl.getByte();
                        for (long i = 0; i < nElements; i++) {
                            copy.setShortFlat((short) (copy.getShortFlat(i) * scalarValue), i);
                        }
                    }
                    case INT16 -> {
                        short scalarValue = scalarImpl.getShort();
                        for (long i = 0; i < nElements; i++) {
                            copy.setShortFlat((short) (copy.getShortFlat(i) * scalarValue), i);
                        }
                    }
                    case INT32 -> {
                        int scalarValue = scalarImpl.getInt();
                        for (long i = 0; i < nElements; i++) {
                            copy.setShortFlat((short) (copy.getShortFlat(i) * scalarValue), i);
                        }
                    }
                    case INT64 -> {
                        long scalarValue = scalarImpl.getLong();
                        for (long i = 0; i < nElements; i++) {
                            copy.setShortFlat((short) (copy.getShortFlat(i) * scalarValue), i);
                        }
                    }
                    case FLOAT32 -> {
                        float scalarValue = scalarImpl.getFloat();
                        for (long i = 0; i < nElements; i++) {
                            copy.setShortFlat((short) (copy.getShortFlat(i) * scalarValue), i);
                        }
                    }
                    case FLOAT64 -> {
                        double scalarValue = scalarImpl.getDouble();
                        for (long i = 0; i < nElements; i++) {
                            copy.setShortFlat((short) (copy.getShortFlat(i) * scalarValue), i);
                        }
                    }
                }
            }
            case INT32 -> {
                switch (scalarDataType) {
                    case INT8 -> {
                        byte scalarValue = scalarImpl.getByte();
                        for (long i = 0; i < nElements; i++) {
                            copy.setIntFlat(copy.getIntFlat(i) * scalarValue, i);
                        }
                    }
                    case INT16 -> {
                        short scalarValue = scalarImpl.getShort();
                        for (long i = 0; i < nElements; i++) {
                            copy.setIntFlat(copy.getIntFlat(i) * scalarValue, i);
                        }
                    }
                    case INT32 -> {
                        int scalarValue = scalarImpl.getInt();
                        for (long i = 0; i < nElements; i++) {
                            copy.setIntFlat(copy.getIntFlat(i) * scalarValue, i);
                        }
                    }
                    case INT64 -> {
                        long scalarValue = scalarImpl.getLong();
                        for (long i = 0; i < nElements; i++) {
                            copy.setIntFlat((int) (copy.getIntFlat(i) * scalarValue), i);
                        }
                    }
                    case FLOAT32 -> {
                        float scalarValue = scalarImpl.getFloat();
                        for (long i = 0; i < nElements; i++) {
                            copy.setIntFlat((int) (copy.getIntFlat(i) * scalarValue), i);
                        }
                    }
                    case FLOAT64 -> {
                        double scalarValue = scalarImpl.getDouble();
                        for (long i = 0; i < nElements; i++) {
                            copy.setIntFlat((int) (copy.getIntFlat(i) * scalarValue), i);
                        }
                    }
                }
            }
            case INT64 -> {
                switch (scalarDataType) {
                    case INT8 -> {
                        byte scalarValue = scalarImpl.getByte();
                        for (long i = 0; i < nElements; i++) {
                            copy.setLongFlat(copy.getLongFlat(i) * scalarValue, i);
                        }
                    }
                    case INT16 -> {
                        short scalarValue = scalarImpl.getShort();
                        for (long i = 0; i < nElements; i++) {
                            copy.setLongFlat(copy.getLongFlat(i) * scalarValue, i);
                        }
                    }
                    case INT32 -> {
                        int scalarValue = scalarImpl.getInt();
                        for (long i = 0; i < nElements; i++) {
                            copy.setLongFlat(copy.getLongFlat(i) * scalarValue, i);
                        }
                    }
                    case INT64 -> {
                        long scalarValue = scalarImpl.getLong();
                        for (long i = 0; i < nElements; i++) {
                            copy.setLongFlat(copy.getLongFlat(i) * scalarValue, i);
                        }
                    }
                    case FLOAT32 -> {
                        float scalarValue = scalarImpl.getFloat();
                        for (long i = 0; i < nElements; i++) {
                            copy.setLongFlat((long) (copy.getLongFlat(i) * scalarValue), i);
                        }
                    }
                    case FLOAT64 -> {
                        double scalarValue = scalarImpl.getDouble();
                        for (long i = 0; i < nElements; i++) {
                            copy.setLongFlat((long) (copy.getLongFlat(i) * scalarValue), i);
                        }
                    }
                }
            }
            case FLOAT32 -> {
                switch (scalarDataType) {
                    case INT8 -> {
                        byte scalarValue = scalarImpl.getByte();
                        for (long i = 0; i < nElements; i++) {
                            copy.setFloatFlat(copy.getFloatFlat(i) * scalarValue, i);
                        }
                    }
                    case INT16 -> {
                        short scalarValue = scalarImpl.getShort();
                        for (long i = 0; i < nElements; i++) {
                            copy.setFloatFlat(copy.getFloatFlat(i) * scalarValue, i);
                        }
                    }
                    case INT32 -> {
                        int scalarValue = scalarImpl.getInt();
                        for (long i = 0; i < nElements; i++) {
                            copy.setFloatFlat(copy.getFloatFlat(i) * scalarValue, i);
                        }
                    }
                    case INT64 -> {
                        long scalarValue = scalarImpl.getLong();
                        for (long i = 0; i < nElements; i++) {
                            copy.setFloatFlat(copy.getFloatFlat(i) * scalarValue, i);
                        }
                    }
                    case FLOAT32 -> {
                        float scalarValue = scalarImpl.getFloat();
                        for (long i = 0; i < nElements; i++) {
                            copy.setFloatFlat(copy.getFloatFlat(i) * scalarValue, i);
                        }
                    }
                    case FLOAT64 -> {
                        double scalarValue = scalarImpl.getDouble();
                        for (long i = 0; i < nElements; i++) {
                            copy.setFloatFlat((float) (copy.getFloatFlat(i) * scalarValue), i);
                        }
                    }
                }
            }
            case FLOAT64 -> {
                switch (scalarDataType) {
                    case INT8 -> {
                        byte scalarValue = scalarImpl.getByte();
                        for (long i = 0; i < nElements; i++) {
                            copy.setDoubleFlat(copy.getDoubleFlat(i) * scalarValue, i);
                        }
                    }
                    case INT16 -> {
                        short scalarValue = scalarImpl.getShort();
                        for (long i = 0; i < nElements; i++) {
                            copy.setDoubleFlat(copy.getDoubleFlat(i) * scalarValue, i);
                        }
                    }
                    case INT32 -> {
                        int scalarValue = scalarImpl.getInt();
                        for (long i = 0; i < nElements; i++) {
                            copy.setDoubleFlat(copy.getDoubleFlat(i) * scalarValue, i);
                        }
                    }
                    case INT64 -> {
                        long scalarValue = scalarImpl.getLong();
                        for (long i = 0; i < nElements; i++) {
                            copy.setDoubleFlat(copy.getDoubleFlat(i) * scalarValue, i);
                        }
                    }
                    case FLOAT32 -> {
                        float scalarValue = scalarImpl.getFloat();
                        for (long i = 0; i < nElements; i++) {
                            copy.setDoubleFlat(copy.getDoubleFlat(i) * scalarValue, i);
                        }
                    }
                    case FLOAT64 -> {
                        double scalarValue = scalarImpl.getDouble();
                        for (long i = 0; i < nElements; i++) {
                            copy.setDoubleFlat(copy.getDoubleFlat(i) * scalarValue, i);
                        }
                    }
                }
            }
        }
        return copy;
    }

    @NotNull
    default ITensor matmul(@NotNull ITensor other) {
        // General matmul (slower)
        long[] otherShape = other.getShape();
        if (otherShape.length != 2) {
            throw new IllegalArgumentException("matmul only supports 2D matrices");
        }
        long[] shape = getShape();
        if (shape.length != 2) {
            throw new IllegalArgumentException("matmul only supports 2D matrices");
        }
        if (shape[1] != otherShape[0]) {
            throw new IllegalArgumentException("matmul: shape mismatch. A.shape[1] != B.shape[0]");
        }
        long[] resultShape = new long[]{shape[0], otherShape[1]};
        DataType ownDataType = getDataType();
        DataType otherDataType = other.getDataType();
        DataType resultDataType = DataType.getLarger(ownDataType, otherDataType);

        SciCoreBackend sc = getSciCore();

        TensorImpl result = sc.createTensor(resultDataType, resultShape);

        long[] index = new long[resultShape.length];
        for (int i = 0; i < resultShape[0]; i++) {
            for (int j = 0; j < resultShape[1]; j++) {
                if (resultDataType.isFloatingPoint()) {
                    double sum = 0;
                    for (int k = 0; k < shape[1]; k++) {
                        index[0] = i;
                        index[1] = k;
                        double aValue = switch (ownDataType) {
                            case INT8 -> getByte(index);
                            case INT16 -> getShort(index);
                            case INT32 -> getInt(index);
                            case INT64 -> getLong(index);
                            case FLOAT32 -> getFloat(index);
                            case FLOAT64 -> getDouble(index);
                        };
                        index[0] = k;
                        index[1] = j;
                        double bValue = switch (otherDataType) {
                            case INT8 -> other.getByte(index);
                            case INT16 -> other.getShort(index);
                            case INT32 -> other.getInt(index);
                            case INT64 -> other.getLong(index);
                            case FLOAT32 -> other.getFloat(index);
                            case FLOAT64 -> other.getDouble(index);
                        };
                        sum += aValue * bValue;
                    }
                    index[0] = i;
                    index[1] = j;
                    switch (resultDataType) {
                        case FLOAT32 -> result.setFloat((float) sum, index);
                        case FLOAT64 -> result.setDouble(sum, index);
                        default -> throw new IllegalStateException("Unexpected data type: " + resultDataType);
                    }
                } else {
                    long sum = 0;
                    for (int k = 0; k < shape[1]; k++) {
                        index[0] = i;
                        index[1] = k;
                        long aValue = switch (ownDataType) {
                            case INT8 -> getByte(index);
                            case INT16 -> getShort(index);
                            case INT32 -> getInt(index);
                            case INT64 -> getLong(index);
                            default -> throw new IllegalStateException("Unexpected data type: " + ownDataType);
                        };
                        index[0] = k;
                        index[1] = j;
                        long bValue = switch (otherDataType) {
                            case INT8 -> other.getByte(index);
                            case INT16 -> other.getShort(index);
                            case INT32 -> other.getInt(index);
                            case INT64 -> other.getLong(index);
                            default -> throw new IllegalStateException("Unexpected data type: " + otherDataType);
                        };
                        sum += aValue * bValue;
                    }
                    index[0] = i;
                    index[1] = j;
                    switch (resultDataType) {
                        case INT8 -> result.setByte((byte) sum, index);
                        case INT16 -> result.setShort((short) sum, index);
                        case INT32 -> result.setInt((int) sum, index);
                        case INT64 -> result.setLong(sum, index);
                        default -> throw new IllegalStateException("Unexpected data type: " + resultDataType);
                    }
                }
            }
        }
        return new Tensor(result, getSciCore());
    }

    void fill(byte i);

    void fill(short i);

    void fill(int i);

    void fill(long i);

    void fill(float i);

    void fill(double i);

    @NotNull
    default ITensor exp() {
        long[] shape = getShape();
        long nElements = ShapeUtils.getNumElements(shape);
        SciCoreBackend sc = getSciCore();
        TensorImpl result = sc.createTensor(getDataType(), shape);
        for (long i = 0; i < nElements; i++) {
            result.setDoubleFlat(Math.exp(getDoubleFlat(i)), i);
        }
        return new Tensor(result, getSciCore());
    }

    @NotNull
    default ITensor softmax(int dimension) {
        ITensor exponentiated = exp();
        ITensor sum = exponentiated.reduceSum(dimension, true);
        return exponentiated.divided(sum);
    }

    @NotNull
    default ITensor divided(@NotNull ITensor other) {
        long[] shapeA = getShape();
        long[] stridesA = ShapeUtils.makeStrides(shapeA);
        long[] shapeB = other.getShape();
        long[] stridesB = ShapeUtils.makeStrides(shapeB);
        long[] outputShape = ShapeUtils.broadcastShapes(shapeA, shapeB);
        long[] stridesOut = ShapeUtils.makeStrides(outputShape);

        SciCoreBackend sc = getSciCore();
        DataType ownDataType = getDataType();
        DataType otherDataType = other.getDataType();
        DataType dataType = DataType.getLarger(ownDataType, otherDataType);
        TensorImpl result = sc.createTensor(dataType, outputShape);

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

            if (dataType.isFloatingPoint()) {
                double a = switch (ownDataType) {
                    case INT8 -> getByteFlat(indexAFlat);
                    case INT16 -> getShortFlat(indexAFlat);
                    case INT32 -> getIntFlat(indexAFlat);
                    case INT64 -> getLongFlat(indexAFlat);
                    case FLOAT32 -> getFloatFlat(indexAFlat);
                    case FLOAT64 -> getDoubleFlat(indexAFlat);
                };
                double b = switch (otherDataType) {
                    case INT8 -> other.getByteFlat(indexBFlat);
                    case INT16 -> other.getShortFlat(indexBFlat);
                    case INT32 -> other.getIntFlat(indexBFlat);
                    case INT64 -> other.getLongFlat(indexBFlat);
                    case FLOAT32 -> other.getFloatFlat(indexBFlat);
                    case FLOAT64 -> other.getDoubleFlat(indexBFlat);
                };
                double aDivB = a / b;
                switch (dataType) {
                    case FLOAT32 -> result.setFloatFlat((float) aDivB, outputIndexFlat);
                    case FLOAT64 -> result.setDoubleFlat(aDivB, outputIndexFlat);
                }
            } else {
                long a = switch (ownDataType) {
                    case INT8 -> getByteFlat(indexAFlat);
                    case INT16 -> getShortFlat(indexAFlat);
                    case INT32 -> getIntFlat(indexAFlat);
                    case INT64 -> getLongFlat(indexAFlat);
                    default -> throw new IllegalStateException("Illegal data type");
                };
                long b = switch (ownDataType) {
                    case INT8 -> other.getByteFlat(indexBFlat);
                    case INT16 -> other.getShortFlat(indexBFlat);
                    case INT32 -> other.getIntFlat(indexBFlat);
                    case INT64 -> other.getLongFlat(indexBFlat);
                    default -> throw new IllegalStateException("Illegal data type");
                };
                long aDivB = a / b;
                switch (dataType) {
                    case INT8 -> result.setByteFlat((byte) aDivB, outputIndexFlat);
                    case INT16 -> result.setShortFlat((byte) aDivB, outputIndexFlat);
                    case INT32 -> result.setIntFlat((byte) aDivB, outputIndexFlat);
                    case INT64 -> result.setLongFlat((byte) aDivB, outputIndexFlat);
                }
            }
        } while (ShapeUtils.incrementIndex(outputShape, outputIndex));
        return new Tensor(result, getSciCore());
    }

    @NotNull
    default ITensor reduceSum(int dimension) {
        return reduceSum(dimension, false);
    }

    @NotNull
    default ITensor reduceSum(int dimension, boolean keepDimensions) {
        // TODO: OPTIMIZE
        SciCoreBackend sc = getSciCore();
        DataType dataType = getDataType();
        long[] shape = getShape();
        if (dimension == -1) {
            TensorImpl result = keepDimensions ? sc.createTensor(dataType, new long[]{1, 1}) : sc.createTensor(dataType, new long[]{1});
            long numElements = ShapeUtils.getNumElements(shape);
            if (dataType.isFloatingPoint()) {
                double sum = 0;
                for (long i = 0; i < numElements; i++) {
                    switch (dataType) {
                        case FLOAT32 -> sum += getFloatFlat(i);
                        case FLOAT64 -> sum += getDoubleFlat(i);
                        default -> throw new IllegalStateException("Unexpected data type: " + dataType);
                    }
                }
                switch (dataType) {
                    case FLOAT32 -> result.setFloatFlat((float) sum, 0);
                    case FLOAT64 -> result.setDoubleFlat(sum, 0);
                    default -> throw new IllegalStateException("Unexpected data type: " + dataType);
                }
            } else {
                long sum = 0;
                for (long i = 0; i < numElements; i++) {
                    switch (dataType) {
                        case INT8 -> sum += getByteFlat(i);
                        case INT16 -> sum += getShortFlat(i);
                        case INT32 -> sum += getIntFlat(i);
                        case INT64 -> sum += getLongFlat(i);
                        default -> throw new IllegalStateException("Unexpected data type: " + dataType);
                    }
                }
                switch (dataType) {
                    case INT8 -> result.setByteFlat((byte) sum, 0);
                    case INT16 -> result.setShortFlat((short) sum, 0);
                    case INT32 -> result.setIntFlat((int) sum, 0);
                    case INT64 -> result.setLongFlat(sum, 0);
                    default -> throw new IllegalStateException("Unexpected data type: " + dataType);
                }
            }
            return new Tensor(result, sc);
        }
        if (dimension < 0 || dimension >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds: " + dimension);
        }
        long[] reducedShape = new long[shape.length - (keepDimensions ? 0 : 1)];
        for (int i = 0; i < shape.length; i++) {
            long dimSize = shape[i];
            if (keepDimensions) {
                if (i == dimension) {
                    dimSize = 1;
                }
                reducedShape[i] = dimSize;
            } else {
                if (i < dimension) {
                    reducedShape[i] = dimSize;
                } else if (i > dimension) {
                    reducedShape[i - 1] = dimSize;
                }
            }
        }

        TensorImpl result = sc.createTensor(dataType, reducedShape);
        long[] completeIndex = new long[shape.length];
        long[] reducedIndex = new long[reducedShape.length];

        while (true) {
            if (dataType.isFloatingPoint()) {
                double sum = 0;
                for (long i = 0; i < shape[dimension]; i++) {
                    completeIndex[dimension] = i;
                    sum += switch (dataType) {
                        case FLOAT32 -> getFloat(completeIndex);
                        case FLOAT64 -> getDouble(completeIndex);
                        default -> throw new IllegalStateException("Unexpected data type: " + dataType);
                    };
                }

                switch (dataType) {
                    case FLOAT32 -> result.setFloat((float) sum, reducedIndex);
                    case FLOAT64 -> result.setDouble(sum, reducedIndex);
                }
            } else {
                long sum = 0;

                for (long i = 0; i < shape[dimension]; i++) {
                    completeIndex[dimension] = i;
                    sum += switch (dataType) {
                        case INT8 -> getByte(completeIndex);
                        case INT16 -> getShort(completeIndex);
                        case INT32 -> getInt(completeIndex);
                        case INT64 -> getLong(completeIndex);
                        default -> throw new IllegalStateException("Unexpected data type: " + dataType);
                    };
                }

                switch (dataType) {
                    case INT8 -> result.setByte((byte) sum, reducedIndex);
                    case INT16 -> result.setShort((short) sum, reducedIndex);
                    case INT32 -> result.setInt((int) sum, reducedIndex);
                    case INT64 -> result.setLong(sum, reducedIndex);
                    default -> throw new IllegalStateException("Unexpected data type: " + dataType);
                }
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
        return new Tensor(result, getSciCore());
    }

    @NotNull ITensorIterator iterator();

    @NotNull SciCoreBackend getSciCore();
}
