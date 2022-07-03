package me.mikex86.scicore;

import me.mikex86.scicore.backend.ScalarImpl;
import me.mikex86.scicore.backend.SciCoreBackend;
import me.mikex86.scicore.backend.TensorImpl;
import me.mikex86.scicore.backend.impl.jvm.JvmTensorImpl;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

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

    @NotNull
    ITensor copy();

    void setContents(@NotNull ITensor tensor);

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

        JvmTensorImpl result = new JvmTensorImpl(resultDataType, resultShape);

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
        TensorImpl result = new JvmTensorImpl(getDataType(), shape);
        for (long i = 0; i < nElements; i++) {
            result.setDoubleFlat(Math.exp(getDoubleFlat(i)), i);
        }
        return new Tensor(result, getSciCore());
    }

    @NotNull
    default ITensor softmax(int dimension) {
        ITensor exponentiated = exp();
        ITensor sum = exponentiated.reduceSum(dimension);
        return exponentiated.divided(sum);
    }

    @NotNull
    default ITensor divided(@NotNull ITensor sum) {
        long[] shape = getShape();
        long[] otherShape = sum.getShape();
        if (ShapeUtils.equals(shape, otherShape)) {
            throw new IllegalArgumentException("Shapes must not be equal");
        }
        long nElements = ShapeUtils.getNumElements(shape);
        DataType ownDataType = getDataType();
        DataType otherDataType = sum.getDataType();
        DataType resultDataType = DataType.getLarger(ownDataType, otherDataType);
        TensorImpl result = new JvmTensorImpl(resultDataType, shape);
        for (long i = 0; i < nElements; i++) {
            if (resultDataType.isFloatingPoint()) {
                double a = switch (ownDataType) {
                    case FLOAT32 -> getFloatFlat(i);
                    case FLOAT64 -> getDoubleFlat(i);
                    default -> throw new IllegalStateException("Unexpected data type: " + ownDataType);
                };
                double b = switch (otherDataType) {
                    case FLOAT32 -> sum.getFloatFlat(i);
                    case FLOAT64 -> sum.getDoubleFlat(i);
                    default -> throw new IllegalStateException("Unexpected data type: " + otherDataType);
                };
                switch (resultDataType) {
                    case FLOAT32 -> result.setFloatFlat((float) (a / b), i);
                    case FLOAT64 -> result.setDoubleFlat(a / b, i);
                }
            } else {
                long a = switch (ownDataType) {
                    case INT8 -> getByteFlat(i);
                    case INT16 -> getShortFlat(i);
                    case INT32 -> getIntFlat(i);
                    case INT64 -> getLongFlat(i);
                    default -> throw new IllegalStateException("Unexpected data type: " + ownDataType);
                };
                long b = switch (otherDataType) {
                    case INT8 -> sum.getByteFlat(i);
                    case INT16 -> sum.getShortFlat(i);
                    case INT32 -> sum.getIntFlat(i);
                    case INT64 -> sum.getLongFlat(i);
                    default -> throw new IllegalStateException("Unexpected data type: " + otherDataType);
                };
                switch (resultDataType) {
                    case INT8 -> result.setByteFlat((byte) (a / b), i);
                    case INT16 -> result.setShortFlat((short) (a / b), i);
                    case INT32 -> result.setIntFlat((int) (a / b), i);
                    case INT64 -> result.setLongFlat(a / b, i);
                    default -> throw new IllegalStateException("Unexpected data type: " + resultDataType);
                }
            }
        }
        return new Tensor(result, getSciCore());
    }

    @NotNull
    default ITensor reduceSum(int dimension) {
        long[] shape = getShape();
        if (dimension < 0 || dimension >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds: " + dimension);
        }
        long[] resultingShape;
        long nNumSums = 1;
        long nElementsToSum = shape[dimension];
        {
            resultingShape = new long[shape.length];
            int j = 0;
            for (int i = 0; i < shape.length; i++) {
                long dimSize = shape[i];
                if (i != dimension) {
                    resultingShape[j++] = dimSize;
                    nNumSums *= dimSize;
                }
            }
            resultingShape = Arrays.copyOf(resultingShape, j);
        }

        DataType dataType = getDataType();
        TensorImpl result = new JvmTensorImpl(dataType, resultingShape);
        for (long s = 0; s < nNumSums; s++) {
            if (dataType.isFloatingPoint()) {
                double sum = 0;
                for (long e = 0; e < nElementsToSum; e++) {
                    long index = s * sumStride + e * elementStride;
                    switch (dataType) {
                        case FLOAT32 -> sum += getFloatFlat(index);
                        case FLOAT64 -> sum += getDoubleFlat(index);
                        default -> throw new IllegalStateException("Unexpected data type: " + dataType);
                    }
                }
                switch (dataType) {
                    case FLOAT32 -> result.setFloatFlat((float) sum, s);
                    case FLOAT64 -> result.setDoubleFlat(sum, s);
                }
            } else {
                long sum = 0;
                for (long e = 0; e < nElementsToSum; e++) {
                    long index = s * sumStride + e * elementStride;
                    switch (dataType) {
                        case INT8 -> sum += getByteFlat(index);
                        case INT16 -> sum += getShortFlat(index);
                        case INT32 -> sum += getIntFlat(index);
                        case INT64 -> sum += getLongFlat(index);
                        default -> throw new IllegalStateException("Unexpected data type: " + dataType);
                    }
                }
                switch (dataType) {
                    case INT8 -> result.setByteFlat((byte) sum, s);
                    case INT16 -> result.setShortFlat((short) sum, s);
                    case INT32 -> result.setIntFlat((int) sum, s);
                    case INT64 -> result.setLongFlat(sum, s);
                    default -> throw new IllegalStateException("Unexpected data type: " + dataType);
                }
            }
        }
        return new Tensor(result, getSciCore());
    }

    @NotNull
    ITensorIterator iterator();

    @NotNull SciCoreBackend getSciCore();
}
