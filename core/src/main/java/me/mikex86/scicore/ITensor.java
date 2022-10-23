package me.mikex86.scicore;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.nio.*;

public interface ITensor extends IValue {

    float EPSILON = 1E-4f;

    @NotNull DataType getDataType();

    long @NotNull [] getShape();

    long @NotNull [] getStrides();

    @NotNull ITensor getView(long @NotNull ... indices);

    @NotNull ITensor getReshapedView(long @NotNull [] shape, long @NotNull [] strides);

    default @NotNull ITensor getReshapedView(long @NotNull [] shape) {
        return getReshapedView(shape, ShapeUtils.makeStrides(shape));
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
            case BOOLEAN -> getBooleanFlat(flatIndex) ? (byte) 1 : (byte) 0;
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
            case BOOLEAN -> setBooleanFlat(value != 0, flatIndex);
        }
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
            case BOOLEAN -> getBoolean(indices) ? (byte) 1 : (byte) 0;
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
            case BOOLEAN -> setBoolean(value != 0, indices);
        }
    }

    default byte elementAsByte() {
        Validator.assertTrue(isScalar(), "Requested element of tensor that is not a scalar!");
        return getAsByteFlat(0);
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
            case BOOLEAN -> getBooleanFlat(flatIndex) ? (short) 1 : (short) 0;
        };
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
            case BOOLEAN -> setBooleanFlat(value != 0, flatIndex);
        }
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
            case BOOLEAN -> getBoolean(indices) ? (short) 1 : (short) 0;
        };
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
            case BOOLEAN -> setBoolean(value != 0, indices);
        }
    }

    default short elementAsShort() {
        Validator.assertTrue(isScalar(), "Requested element of tensor that is not a scalar!");
        return getAsShortFlat(0);
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
            case BOOLEAN -> getBooleanFlat(flatIndex) ? 1 : 0;
        };
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
            case BOOLEAN -> setBooleanFlat(value != 0, flatIndex);
        }
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
            case BOOLEAN -> getBoolean(indices) ? 1 : 0;
        };
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

    default int elementAsInt() {
        Validator.assertTrue(isScalar(), "Requested element of tensor that is not a scalar!");
        return getAsIntFlat(0);
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
            case BOOLEAN -> getBooleanFlat(flatIndex) ? 1 : 0;
        };
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
            case BOOLEAN -> setBooleanFlat(value != 0, flatIndex);
        }
    }

    long getLong(long @NotNull ... indices);

    void setLong(long value, long @NotNull ... indices);

    default long getAsLong(long @NotNull ... indices) {
        return switch (getDataType()) {
            case INT8 -> getByte(indices);
            case INT16 -> getShort(indices);
            case INT32 -> getInt(indices);
            case INT64 -> getLong(indices);
            case FLOAT32 -> (long) getFloat(indices);
            case FLOAT64 -> (long) getDouble(indices);
            case BOOLEAN -> getBoolean(indices) ? 1 : 0;
        };
    }

    default void setByLong(long value, long @NotNull ... indices) {
        switch (getDataType()) {
            case INT8 -> setByte((byte) value, indices);
            case INT16 -> setShort((short) value, indices);
            case INT32 -> setInt((int) value, indices);
            case INT64 -> setLong(value, indices);
            case FLOAT32 -> setFloat(value, indices);
            case FLOAT64 -> setDouble(value, indices);
            case BOOLEAN -> setBoolean(value != 0, indices);
        }
    }

    default long elementAsLong() {
        Validator.assertTrue(isScalar(), "Requested element of tensor that is not a scalar!");
        return getAsLongFlat(0);
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
            case BOOLEAN -> getBooleanFlat(flatIndex) ? 1 : 0;
        };
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
            case BOOLEAN -> setBooleanFlat(value != 0, flatIndex);
        }
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
            case BOOLEAN -> getBoolean(indices) ? 1 : 0;
        };
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
            case BOOLEAN -> setBoolean(value != 0, indices);
        }
    }

    default float elementAsFloat() {
        Validator.assertTrue(isScalar(), "Requested element of tensor that is not a scalar!");
        return getAsFloatFlat(0);
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
            case BOOLEAN -> getBooleanFlat(flatIndex) ? 1 : 0;
        };
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
            case BOOLEAN -> setBooleanFlat(value != 0, flatIndex);
        }
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
            case BOOLEAN -> getBoolean(indices) ? 1 : 0;
        };
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
            case BOOLEAN -> setBoolean(value != 0, indices);
        }
    }

    default double elementAsDouble() {
        Validator.assertTrue(isScalar(), "Requested element of tensor that is not a scalar!");
        return getAsDoubleFlat(0);
    }

    boolean getBooleanFlat(long flatIndex);

    default boolean getAsBooleanFlat(long flatIndex) {
        return switch (getDataType()) {
            case INT8 -> getByteFlat(flatIndex) != 0;
            case INT16 -> getShortFlat(flatIndex) != 0;
            case INT32 -> getIntFlat(flatIndex) != 0;
            case INT64 -> getLongFlat(flatIndex) != 0;
            case FLOAT32 -> getFloatFlat(flatIndex) != 0.0;
            case FLOAT64 -> getDoubleFlat(flatIndex) != 0.0;
            case BOOLEAN -> getBooleanFlat(flatIndex);
        };
    }

    void setBooleanFlat(boolean value, long flatIndex);

    default void setByBooleanFlat(boolean value, long flatIndex) {
        switch (getDataType()) {
            case INT8 -> setByteFlat(value ? (byte) 1 : (byte) 0, flatIndex);
            case INT16 -> setShortFlat(value ? (short) 1 : (short) 0, flatIndex);
            case INT32 -> setIntFlat(value ? 1 : 0, flatIndex);
            case INT64 -> setLongFlat(value ? 1 : 0, flatIndex);
            case FLOAT32 -> setFloatFlat(value ? 1.0f : 0.0f, flatIndex);
            case FLOAT64 -> setDoubleFlat(value ? 1.0 : 0.0, flatIndex);
            case BOOLEAN -> setBooleanFlat(value, flatIndex);
        }
    }

    boolean getBoolean(long @NotNull ... indices);

    default boolean getAsBoolean(long @NotNull ... indices) {
        return switch (getDataType()) {
            case INT8 -> getByte(indices) != 0;
            case INT16 -> getShort(indices) != 0;
            case INT32 -> getInt(indices) != 0;
            case INT64 -> getLong(indices) != 0;
            case FLOAT32 -> getFloat(indices) != 0.0;
            case FLOAT64 -> getDouble(indices) != 0.0;
            case BOOLEAN -> getBoolean(indices);
        };
    }

    void setBoolean(boolean value, long @NotNull ... indices);

    default void setByBoolean(boolean value, long @NotNull ... indices) {
        switch (getDataType()) {
            case INT8 -> setByte(value ? (byte) 1 : (byte) 0, indices);
            case INT16 -> setShort(value ? (short) 1 : (short) 0, indices);
            case INT32 -> setInt(value ? 1 : 0, indices);
            case INT64 -> setLong(value ? 1 : 0, indices);
            case FLOAT32 -> setFloat(value ? 1.0f : 0.0f, indices);
            case FLOAT64 -> setDouble(value ? 1.0 : 0.0, indices);
            case BOOLEAN -> setBoolean(value, indices);
        }
    }

    default boolean elementAsBoolean() {
        Validator.assertTrue(isScalar(), "Requested element of tensor that is not a scalar!");
        return getAsBooleanFlat(0);
    }

    /**
     * Returns the element, the only value stored in a scalar tensor.
     *
     * @param typeClass the type class of the element.
     * @throws IllegalArgumentException if the tensor is not a scalar, or the requested type is not the same as the tensor's type.
     */
    @SuppressWarnings("unchecked")
    default <T> T element(@NotNull Class<T> typeClass) {
        Validator.assertTrue(isScalar(), "Requested element of tensor that is not a scalar!");
        Object value = switch (getDataType()) {
            case INT8 -> getByteFlat(0);
            case INT16 -> getShortFlat(0);
            case INT32 -> getIntFlat(0);
            case INT64 -> getLongFlat(0);
            case FLOAT32 -> getFloatFlat(0);
            case FLOAT64 -> getDoubleFlat(0);
            case BOOLEAN -> getBooleanFlat(0);
        };
        Validator.assertTrue(typeClass.isInstance(value), "Requested element type is not compatible with tensor data type");
        return (T) value;
    }

    @NotNull ITensor copy();

    void setContents(@NotNull ITensor tensor);

    void setContents(@NotNull ByteBuffer buffer);

    void setContents(@NotNull ShortBuffer buffer);

    void setContents(@NotNull IntBuffer buffer);

    void setContents(@NotNull LongBuffer buffer);

    void setContents(@NotNull FloatBuffer buffer);

    void setContents(@NotNull DoubleBuffer buffer);

    void setContents(boolean @NotNull [] buffer);

    void setContents(long @NotNull [] index, @NotNull ITensor tensor);

    long getNumberOfElements();

    @NotNull ITensor matmul(@NotNull ITensor other);

    @NotNull ITensor matmul(@NotNull ITensor other, boolean transposeSelf, boolean transposeOther);

    @NotNull ITensor divide(@NotNull ITensor other);

    @NotNull ITensor divide(byte value);

    @NotNull ITensor divide(short value);

    @NotNull ITensor divide(int value);

    @NotNull ITensor divide(long value);

    @NotNull ITensor divide(float value);

    @NotNull ITensor divide(double value);

    @NotNull ITensor plus(byte value);

    @NotNull ITensor plus(short value);

    @NotNull ITensor plus(int value);

    @NotNull ITensor plus(long value);

    @NotNull ITensor plus(float value);

    @NotNull ITensor plus(double value);

    @NotNull ITensor plus(@NotNull ITensor other);

    void add(byte value);

    void add(short value);

    void add(int value);

    void add(long value);

    void add(float value);

    void add(double value);

    void add(@NotNull ITensor other);

    @NotNull ITensor minus(byte value);

    @NotNull ITensor minus(short value);

    @NotNull ITensor minus(int value);

    @NotNull ITensor minus(long value);

    @NotNull ITensor minus(float value);

    @NotNull ITensor minus(double value);

    @NotNull ITensor minus(@NotNull ITensor other);

    void subtract(byte value);

    void subtract(short value);

    void subtract(int value);

    void subtract(long value);

    void subtract(float value);

    void subtract(double value);

    void subtract(@NotNull ITensor other);

    void fill(byte value);

    void fill(short value);

    void fill(int value);

    void fill(long value);

    void fill(float value);

    void fill(double value);

    void fill(boolean value);

    @NotNull ITensor exp();

    @NotNull ITensor softmax(int dimension);

    @NotNull ITensor reduceSum(int dimension);

    @NotNull ITensor reduceSum(int dimension, boolean keepDimensions);

    @NotNull ITensor transpose();

    @Override
    boolean equals(Object other);

    @NotNull ISciCoreBackend getSciCoreBackend();

    default boolean isScalar() {
        return ShapeUtils.isScalar(getShape());
    }

    @NotNull ITensor pow(byte exponent);

    @NotNull ITensor pow(short exponent);

    @NotNull ITensor pow(int exponent);

    @NotNull ITensor pow(long exponent);

    @NotNull ITensor pow(float exponent);

    @NotNull ITensor pow(double exponent);

    @NotNull ITensor pow(@NotNull ITensor exponent);

    /**
     * Multiplies this tensor by the other tensor either:
     * Element-wise multiplication if the tensors have the same shape,
     * or dimension-wise multiplication if the tensors have different shapes that are broad-castable.
     *
     * @param other the other tensor.
     * @return the result of the multiplication.
     */
    @NotNull ITensor multiply(@NotNull ITensor other);

    @NotNull ITensor multiply(byte value);

    @NotNull ITensor multiply(short value);

    @NotNull ITensor multiply(int value);

    @NotNull ITensor multiply(long value);

    @NotNull ITensor multiply(float value);

    @NotNull ITensor multiply(double value);

    @NotNull ITensor relu();

    @NotNull ITensor sigmoid();

    @NotNull ITensor argmax(int dimension);

    @NotNull ITensor to(@NotNull ISciCoreBackend backend);

    /**
     * @return a direct memory handle containing the contents of the tensor. Can be a reference to the tensor's internal memory, or a copy. Only copies can be freed.
     * @see DirectMemoryHandle#canFree()
     */
    @NotNull DirectMemoryHandle getContentsAsDirectMemory();

    /**
     * @param startFlatIndex the flat index of the first element to retrieve
     * @param endFlatIndex   the flat index of the last element to retrieve (exclusive)
     * @return a direct memory handle containing the tensor's data in the specified interval. Can be a reference to the tensor's internal memory, or a copy. Only copies can be freed.
     * @see DirectMemoryHandle#canFree()
     */
    @NotNull DirectMemoryHandle getContentsAsDirectMemory(long startFlatIndex, long endFlatIndex);

    // TODO: USE getIfIsType EVERYWHERE INSTEAD OF INSTANCEOF
    default <T extends ITensor> @Nullable T getIfIsType(@NotNull Class<T> typeClass) {
        return typeClass.isInstance(this) ? typeClass.cast(this) : null;
    }

    @NotNull ITensor cast(@NotNull DataType dataType);

    /**
     * @param tensor the tensor to check
     * @return true, if the tensor is the same instance as this tensor. Lazy tensors that wrap the same tensor are not considered equal.
     */
    boolean isSame(@NotNull ITensor tensor);
}
