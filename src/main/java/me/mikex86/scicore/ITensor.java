package me.mikex86.scicore;

import me.mikex86.scicore.backend.ScalarImpl;
import me.mikex86.scicore.backend.SciCoreBackend;
import me.mikex86.scicore.backend.ITensorImpl;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;

public interface ITensor extends IValue {

    float EPSILON = 1E-5f;

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

    long @NotNull [] getStrides();

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

    default byte getAsByte(long @NotNull ... indices) {
        return switch (getDataType()) {
            case INT8 -> getByte(indices);
            case INT16 -> (byte) getShort(indices);
            case INT32 -> (byte) getInt(indices);
            case INT64 -> (byte) getLong(indices);
            case FLOAT32 -> (byte) getFloat(indices);
            case FLOAT64 -> (byte) getDouble(indices);
        };
    }

    short getShort(long @NotNull ... indices);

    default short getAsShort(long @NotNull ... indices) {
        return switch (getDataType()) {
            case INT8 -> getByte(indices);
            case INT16 -> getShort(indices);
            case INT32 -> (short) getInt(indices);
            case INT64 -> (short) getLong(indices);
            case FLOAT32 -> (short) getFloat(indices);
            case FLOAT64 -> (short) getDouble(indices);
        };
    }

    int getInt(long @NotNull ... indices);

    default int getAsInt(long @NotNull ... indices) {
        return switch (getDataType()) {
            case INT8 -> getByte(indices);
            case INT16 -> getShort(indices);
            case INT32 -> getInt(indices);
            case INT64 -> (int) getLong(indices);
            case FLOAT32 -> (int) getFloat(indices);
            case FLOAT64 -> (int) getDouble(indices);
        };
    }

    long getLong(long @NotNull ... indices);

    default long getAsLong(long @NotNull ... indices) {
        return switch (getDataType()) {
            case INT8 -> getByte(indices);
            case INT16 -> getShort(indices);
            case INT32 -> getInt(indices);
            case INT64 -> getLong(indices);
            case FLOAT32 -> (long) getFloat(indices);
            case FLOAT64 -> (long) getDouble(indices);
        };
    }

    float getFloat(long @NotNull ... indices);

    default float getAsFloat(long @NotNull ... indices) {
        return switch (getDataType()) {
            case INT8 -> getByte(indices);
            case INT16 -> getShort(indices);
            case INT32 -> getInt(indices);
            case INT64 -> getLong(indices);
            case FLOAT32 -> getFloat(indices);
            case FLOAT64 -> (float) getDouble(indices);
        };
    }

    double getDouble(long @NotNull ... indices);

    default double getAsDouble(long @NotNull ... indices) {
        return switch (getDataType()) {
            case INT8 -> getByte(indices);
            case INT16 -> getShort(indices);
            case INT32 -> getInt(indices);
            case INT64 -> getLong(indices);
            case FLOAT32 -> getFloat(indices);
            case FLOAT64 -> getDouble(indices);
        };
    }

    void setByte(byte value, long @NotNull ... indices);

    default void setByByte(byte value, long @NotNull ... indices) {
        switch (getDataType()) {
            case INT8 -> setByte(value, indices);
            case INT16 -> setShort(value, indices);
            case INT32 -> setInt(value, indices);
            case INT64 -> setLong(value, indices);
            case FLOAT32 -> setFloat(value, indices);
            case FLOAT64 -> setDouble(value, indices);
        }
    }

    void setShort(short value, long @NotNull ... indices);

    default void setByShort(short value, long @NotNull ... indices) {
        switch (getDataType()) {
            case INT8 -> setByte((byte) value, indices);
            case INT16 -> setShort(value, indices);
            case INT32 -> setInt(value, indices);
            case INT64 -> setLong(value, indices);
            case FLOAT32 -> setFloat(value, indices);
            case FLOAT64 -> setDouble(value, indices);
        }
    }

    void setInt(int value, long @NotNull ... indices);

    default void setByInt(int value, long @NotNull ... indices) {
        switch (getDataType()) {
            case INT8 -> setByte((byte) value, indices);
            case INT16 -> setShort((short) value, indices);
            case INT32 -> setInt(value, indices);
            case INT64 -> setLong(value, indices);
            case FLOAT32 -> setFloat(value, indices);
            case FLOAT64 -> setDouble(value, indices);
        }
    }

    void setLong(long value, long @NotNull ... indices);

    default void setByLong(long value, long @NotNull ... indices) {
        switch (getDataType()) {
            case INT8 -> setByte((byte) value, indices);
            case INT16 -> setShort((short) value, indices);
            case INT32 -> setInt((int) value, indices);
            case INT64 -> setLong(value, indices);
            case FLOAT32 -> setFloat(value, indices);
            case FLOAT64 -> setDouble(value, indices);
        }
    }

    void setFloat(float value, long @NotNull ... indices);

    default void setByFloat(float value, long @NotNull ... indices) {
        switch (getDataType()) {
            case INT8 -> setByte((byte) value, indices);
            case INT16 -> setShort((short) value, indices);
            case INT32 -> setInt((int) value, indices);
            case INT64 -> setLong((long) value, indices);
            case FLOAT32 -> setFloat(value, indices);
            case FLOAT64 -> setDouble(value, indices);
        }
    }

    void setDouble(double value, long @NotNull ... indices);

    default void setByDouble(double value, long @NotNull ... indices) {
        switch (getDataType()) {
            case INT8 -> setByte((byte) value, indices);
            case INT16 -> setShort((short) value, indices);
            case INT32 -> setInt((int) value, indices);
            case INT64 -> setLong((long) value, indices);
            case FLOAT32 -> setFloat((float) value, indices);
            case FLOAT64 -> setDouble(value, indices);
        }
    }

    byte getByteFlat(long flatIndex);

    default byte getAsByteFlat(long flatIndex) {
        return switch (getDataType()) {
            case INT8 -> getByteFlat(flatIndex);
            case INT16 -> (byte) getShortFlat(flatIndex);
            case INT32 -> (byte) getIntFlat(flatIndex);
            case INT64 -> (byte) getLongFlat(flatIndex);
            case FLOAT32 -> (byte) getFloatFlat(flatIndex);
            case FLOAT64 -> (byte) getDoubleFlat(flatIndex);
        };
    }

    short getShortFlat(long flatIndex);

    default short getAsShortFlat(long flatIndex) {
        return switch (getDataType()) {
            case INT8 -> getByteFlat(flatIndex);
            case INT16 -> getShortFlat(flatIndex);
            case INT32 -> (short) getIntFlat(flatIndex);
            case INT64 -> (short) getLongFlat(flatIndex);
            case FLOAT32 -> (short) getFloatFlat(flatIndex);
            case FLOAT64 -> (short) getDoubleFlat(flatIndex);
        };
    }

    int getIntFlat(long flatIndex);

    default int getAsIntFlat(long flatIndex) {
        return switch (getDataType()) {
            case INT8 -> getByteFlat(flatIndex);
            case INT16 -> getShortFlat(flatIndex);
            case INT32 -> getIntFlat(flatIndex);
            case INT64 -> (int) getLongFlat(flatIndex);
            case FLOAT32 -> (int) getFloatFlat(flatIndex);
            case FLOAT64 -> (int) getDoubleFlat(flatIndex);
        };
    }

    long getLongFlat(long flatIndex);

    default long getAsLongFlat(long flatIndex) {
        return switch (getDataType()) {
            case INT8 -> getByteFlat(flatIndex);
            case INT16 -> getShortFlat(flatIndex);
            case INT32 -> getIntFlat(flatIndex);
            case INT64 -> getLongFlat(flatIndex);
            case FLOAT32 -> (long) getFloatFlat(flatIndex);
            case FLOAT64 -> (long) getDoubleFlat(flatIndex);
        };
    }

    float getFloatFlat(long flatIndex);

    default float getAsFloatFlat(long flatIndex) {
        return switch (getDataType()) {
            case INT8 -> getByteFlat(flatIndex);
            case INT16 -> getShortFlat(flatIndex);
            case INT32 -> getIntFlat(flatIndex);
            case INT64 -> getLongFlat(flatIndex);
            case FLOAT32 -> getFloatFlat(flatIndex);
            case FLOAT64 -> (float) getDoubleFlat(flatIndex);
        };
    }

    double getDoubleFlat(long flatIndex);

    default double getAsDoubleFlat(long flatIndex) {
        return switch (getDataType()) {
            case INT8 -> getByteFlat(flatIndex);
            case INT16 -> getShortFlat(flatIndex);
            case INT32 -> getIntFlat(flatIndex);
            case INT64 -> getLongFlat(flatIndex);
            case FLOAT32 -> getFloatFlat(flatIndex);
            case FLOAT64 -> getDoubleFlat(flatIndex);
        };
    }

    void setByteFlat(byte value, long flatIndex);

    default void setByByteFlat(byte value, long flatIndex) {
        switch (getDataType()) {
            case INT8 -> setByteFlat(value, flatIndex);
            case INT16 -> setShortFlat(value, flatIndex);
            case INT32 -> setIntFlat(value, flatIndex);
            case INT64 -> setLongFlat(value, flatIndex);
            case FLOAT32 -> setFloatFlat(value, flatIndex);
            case FLOAT64 -> setDoubleFlat(value, flatIndex);
        }
    }

    void setShortFlat(short value, long flatIndex);

    default void setByShortFlat(short value, long flatIndex) {
        switch (getDataType()) {
            case INT8 -> setByteFlat((byte) value, flatIndex);
            case INT16 -> setShortFlat(value, flatIndex);
            case INT32 -> setIntFlat(value, flatIndex);
            case INT64 -> setLongFlat(value, flatIndex);
            case FLOAT32 -> setFloatFlat(value, flatIndex);
            case FLOAT64 -> setDoubleFlat(value, flatIndex);
        }
    }

    void setIntFlat(int value, long flatIndex);

    default void setByIntFlat(int value, long flatIndex) {
        switch (getDataType()) {
            case INT8 -> setByteFlat((byte) value, flatIndex);
            case INT16 -> setShortFlat((short) value, flatIndex);
            case INT32 -> setIntFlat(value, flatIndex);
            case INT64 -> setLongFlat(value, flatIndex);
            case FLOAT32 -> setFloatFlat(value, flatIndex);
            case FLOAT64 -> setDoubleFlat(value, flatIndex);
        }
    }

    void setLongFlat(long value, long flatIndex);

    default void setByLongFlat(long value, long flatIndex) {
        switch (getDataType()) {
            case INT8 -> setByteFlat((byte) value, flatIndex);
            case INT16 -> setShortFlat((short) value, flatIndex);
            case INT32 -> setIntFlat((int) value, flatIndex);
            case INT64 -> setLongFlat(value, flatIndex);
            case FLOAT32 -> setFloatFlat(value, flatIndex);
            case FLOAT64 -> setDoubleFlat(value, flatIndex);
        }
    }

    void setFloatFlat(float value, long flatIndex);

    default void setByFloatFlat(float value, long flatIndex) {
        switch (getDataType()) {
            case INT8 -> setByteFlat((byte) value, flatIndex);
            case INT16 -> setShortFlat((short) value, flatIndex);
            case INT32 -> setIntFlat((int) value, flatIndex);
            case INT64 -> setLongFlat((long) value, flatIndex);
            case FLOAT32 -> setFloatFlat(value, flatIndex);
            case FLOAT64 -> setDoubleFlat(value, flatIndex);
        }
    }

    void setDoubleFlat(double value, long flatIndex);

    default void setByDoubleFlat(double value, long flatIndex) {
        switch (getDataType()) {
            case INT8 -> setByteFlat((byte) value, flatIndex);
            case INT16 -> setShortFlat((short) value, flatIndex);
            case INT32 -> setIntFlat((int) value, flatIndex);
            case INT64 -> setLongFlat((long) value, flatIndex);
            case FLOAT32 -> setFloatFlat((float) value, flatIndex);
            case FLOAT64 -> setDoubleFlat(value, flatIndex);
        }
    }

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

        ITensorImpl result = sc.createTensor(resultDataType, resultShape);
        ITensor resultTensor = new Tensor(result, sc);

        long[] index = new long[resultShape.length];
        for (int i = 0; i < resultShape[0]; i++) {
            for (int j = 0; j < resultShape[1]; j++) {
                if (resultDataType.isFloatingPoint()) {
                    double sum = 0;
                    for (int k = 0; k < shape[1]; k++) {
                        index[0] = i;
                        index[1] = k;
                        double aValue = getAsDouble(index);
                        index[0] = k;
                        index[1] = j;
                        double bValue = other.getAsDouble(index);
                        sum += aValue * bValue;
                    }
                    index[0] = i;
                    index[1] = j;
                    resultTensor.setByDouble(sum, index);
                } else {
                    long sum = 0;
                    for (int k = 0; k < shape[1]; k++) {
                        index[0] = i;
                        index[1] = k;
                        long aValue = getAsLong(index);
                        index[0] = k;
                        index[1] = j;
                        long bValue = other.getAsLong(index);
                        sum += aValue * bValue;
                    }
                    index[0] = i;
                    index[1] = j;
                    resultTensor.setByLong(sum, index);
                }
            }
        }
        return resultTensor;
    }

    void fill(byte i);

    void fill(short i);

    void fill(int i);

    void fill(long i);

    void fill(float i);

    void fill(double i);

    @NotNull
    ITensor exp();

    @NotNull
    default ITensor softmax(int dimension) {
        ITensor exponentiated = exp();
        ITensor sum = exponentiated.reduceSum(dimension, true);
        return exponentiated.divided(sum);
    }

    @NotNull
    default ITensor divided(@NotNull ITensor other) {
        long[] shapeA = getShape();
        long[] stridesA = getStrides();
        long[] shapeB = other.getShape();
        long[] stridesB = other.getStrides();
        long[] outputShape = ShapeUtils.broadcastShapes(shapeA, shapeB);
        long[] stridesOut = ShapeUtils.makeStrides(outputShape);

        SciCoreBackend sc = getSciCore();
        DataType ownDataType = getDataType();
        DataType otherDataType = other.getDataType();
        DataType dataType = DataType.getLarger(ownDataType, otherDataType);
        ITensorImpl result = sc.createTensor(dataType, outputShape);
        ITensor resultTensor = new Tensor(result, sc);

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
                double a = getAsDoubleFlat(indexAFlat);
                double b = other.getAsDoubleFlat(indexBFlat);
                double aDivB = a / b;
                resultTensor.setByDoubleFlat(aDivB, outputIndexFlat);
            } else {
                long a = getAsLongFlat(indexAFlat);
                long b = other.getAsLongFlat(indexBFlat);
                long aDivB = a / b;
                resultTensor.setByLongFlat(aDivB, outputIndexFlat);
            }
        } while (ShapeUtils.incrementIndex(outputShape, outputIndex));
        return resultTensor;
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
            ITensorImpl result = keepDimensions ? sc.createTensor(dataType, new long[]{1, 1}) : sc.createTensor(dataType, new long[]{1});
            ITensor resultTensor = new Tensor(result, sc);
            long numElements = ShapeUtils.getNumElements(shape);
            if (dataType.isFloatingPoint()) {
                double sum = 0;
                for (long i = 0; i < numElements; i++) {
                    sum += getAsDoubleFlat(i);
                }
                resultTensor.setByDoubleFlat(sum, 0);
            } else {
                long sum = 0;
                for (long i = 0; i < numElements; i++) {
                    sum += getAsLongFlat(i);
                }
                resultTensor.setByLongFlat(sum, 0);
            }
            return resultTensor;
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

        ITensorImpl result = sc.createTensor(dataType, reducedShape);
        ITensor resultTensor = new Tensor(result, sc);

        long[] completeIndex = new long[shape.length];
        long[] reducedIndex = new long[reducedShape.length];

        while (true) {
            if (dataType.isFloatingPoint()) {
                double sum = 0;
                for (long i = 0; i < shape[dimension]; i++) {
                    completeIndex[dimension] = i;
                    sum += getAsDouble(completeIndex);
                }

                resultTensor.setByDouble(sum, reducedIndex);
            } else {
                long sum = 0;

                for (long i = 0; i < shape[dimension]; i++) {
                    completeIndex[dimension] = i;
                    sum += getAsLong(completeIndex);
                }

                resultTensor.setLong(sum, reducedIndex);
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
        return resultTensor;
    }

    @Override
    boolean equals(Object other);

    @NotNull ITensorIterator iterator();

    @NotNull SciCoreBackend getSciCore();
}
