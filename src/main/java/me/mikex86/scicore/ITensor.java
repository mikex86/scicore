package me.mikex86.scicore;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.op.IGraphRecorder;
import me.mikex86.scicore.op.OperationType;
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
    default ITensor matmul(@NotNull ITensor other) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.MATMUL, this, other);
    }

    @NotNull
    default ITensor divided(@NotNull ITensor other) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.DIVIDED, this, other);
    }

    @NotNull
    default ITensor plus(@NotNull ITensor other) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.PLUS, this, other);
    }

    void fill(byte i);

    void fill(short i);

    void fill(int i);

    void fill(long i);

    void fill(float i);

    void fill(double i);

    @NotNull
    default ITensor exp() {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.EXP, this);
    }

    @NotNull
    default ITensor softmax(int dimension) {
        ITensor exponentiated = exp();
        ITensor sum = exponentiated.reduceSum(dimension, true);
        return exponentiated.divided(sum);
    }

    @NotNull
    default ITensor reduceSum(int dimension) {
        return reduceSum(dimension, false);
    }

    @NotNull
    default ITensor reduceSum(int dimension, boolean keepDimensions) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.REDUCE_SUM, this, dimension, keepDimensions);
    }

    @Override
    boolean equals(Object other);

    @NotNull ITensorIterator iterator();

    @NotNull ISciCoreBackend getSciCoreBackend();

    default boolean isScalar() {
        return getNumberOfElements() == 1;
    }
}
