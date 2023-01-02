package me.mikex86.scicore.tensor;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import me.mikex86.scicore.utils.dispose.IDisposable;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.*;
import java.nio.*;
import java.util.Arrays;
import java.util.List;

public interface ITensor extends IValue, IDisposable, AutoCloseable {

    float EPSILON = 1E-3f;

    long getId();

    @NotNull DataType getDataType();

    long @NotNull [] getShape();

    long @NotNull [] getStrides();

    @NotNull ITensor getView(long @NotNull ... indices);

    @NotNull ITensor view(long @NotNull [] shape, long @NotNull [] strides);

    default @NotNull ITensor view(long @NotNull ... shape) {
        shape = Arrays.copyOf(shape, shape.length);
        long numElements = getNumberOfElements();
        boolean hasInferredDimension = false;
        int dimensionToInfer = -1;
        long numOtherElements = 1;
        for (int i = 0; i < shape.length; i++) {
            long l = shape[i];
            if (l == -1) {
                if (hasInferredDimension) {
                    throw new IllegalArgumentException("Only one dimension can be inferred");
                }
                hasInferredDimension = true;
                dimensionToInfer = i;
            } else {
                numOtherElements *= l;
            }
        }
        if (dimensionToInfer != -1) {
            long inferredDimension = numElements / numOtherElements;
            shape[dimensionToInfer] = inferredDimension;
        }
        return view(shape, ShapeUtils.makeStrides(shape));
    }

    @NotNull ITensor concat(@NotNull ITensor tensor, int dim);

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

    @NotNull
    default ITensor broadcast(long... targetShape) {
        long[] shape = getShape();
        Validator.assertTrue(targetShape.length == shape.length, "The number of broadcast dimensions must match the tensor rank");
        boolean containsInfer = false;
        for (int i = 0; i < targetShape.length; i++) {
            if (targetShape[i] == -1) {
                Validator.assertTrue(!containsInfer, "Only one dimension can be inferred");
                targetShape[i] = shape[i];
                containsInfer = true;
                continue;
            }
            Validator.assertTrue(targetShape[i] == 1 || shape[i] == 1 || targetShape[i] == shape[i], "The broadcast dimensions must be either 1 or the same as the tensor shape");
        }
        try (ITensor one = getSciCoreBackend().createTensor(getDataType(), targetShape)) {
            one.fill(1);
            return one.multiply(this); // TODO: handle this with in-place op
        }
    }

    @NotNull ITensor copy();

    void setContentsWithOffset(long startFlatIndex, @NotNull ITensor tensor);

    default void setContents(@NotNull ITensor tensor) {
        setContentsWithOffset(0, tensor);
    }

    void setContentsWithOffset(long startFlatIndex, @NotNull ByteBuffer buffer);

    default void setContents(@NotNull ByteBuffer buffer) {
        setContentsWithOffset(0, buffer);
    }

    void setContentsWithOffset(long startFlatIndex, @NotNull ShortBuffer buffer);

    default void setContents(@NotNull ShortBuffer buffer) {
        setContentsWithOffset(0, buffer);
    }

    void setContentsWithOffset(long startFlatIndex, @NotNull IntBuffer buffer);

    default void setContents(@NotNull IntBuffer buffer) {
        setContentsWithOffset(0, buffer);
    }

    void setContentsWithOffset(long startFlatIndex, @NotNull LongBuffer buffer);

    default void setContents(@NotNull LongBuffer buffer) {
        setContentsWithOffset(0, buffer);
    }

    void setContentsWithOffset(long startFlatIndex, @NotNull FloatBuffer buffer);

    default void setContents(@NotNull FloatBuffer buffer) {
        setContentsWithOffset(0, buffer);
    }

    void setContentsWithOffset(long startFlatIndex, @NotNull DoubleBuffer buffer);

    default void setContents(@NotNull DoubleBuffer buffer) {
        setContentsWithOffset(0, buffer);
    }

    void setContentsWithOffset(long startFlatIndex, boolean @NotNull [] buffer);

    default void setContents(boolean @NotNull [] buffer) {
        setContentsWithOffset(0, buffer);
    }

    default void setContents(long @NotNull [] index, @NotNull ITensor tensor) {
        long flatIndex = ShapeUtils.getFlatIndex(index, getStrides());
        long[] shape = getShape();
        long[] subTensorShape = Arrays.copyOfRange(shape, index.length, shape.length);
        long numElements = ShapeUtils.getNumElements(subTensorShape);
        if (tensor.getNumberOfElements() != numElements) {
            throw new IllegalArgumentException("Dimensions of destination tensor do not match with source tensor in setContents(index, tensor)");
        }
        setContentsWithOffset(flatIndex, tensor);
    }

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


    @NotNull ITensor leftDivide(@NotNull ITensor other);

    @NotNull ITensor leftDivide(byte value);

    @NotNull ITensor leftDivide(short value);

    @NotNull ITensor leftDivide(int value);

    @NotNull ITensor leftDivide(long value);

    @NotNull ITensor leftDivide(float value);

    @NotNull ITensor leftDivide(double value);

    @NotNull
    default ITensor div(@NotNull ITensor other) {
        return divide(other);
    }

    @NotNull
    default ITensor div(byte value) {
        return divide(value);
    }

    @NotNull
    default ITensor div(short value) {
        return divide(value);
    }

    @NotNull
    default ITensor div(int value) {
        return divide(value);
    }

    @NotNull
    default ITensor div(long value) {
        return divide(value);
    }

    @NotNull
    default ITensor div(float value) {
        return divide(value);
    }

    @NotNull
    default ITensor div(double value) {
        return divide(value);
    }

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

    @NotNull ITensor leftMinus(byte value);

    @NotNull ITensor leftMinus(short value);

    @NotNull ITensor leftMinus(int value);

    @NotNull ITensor leftMinus(long value);

    @NotNull ITensor leftMinus(float value);

    @NotNull ITensor leftMinus(double value);

    void subtract(byte value);

    void subtract(short value);

    void subtract(int value);

    void subtract(long value);

    void subtract(float value);

    void subtract(double value);

    void subtract(@NotNull ITensor other);

    void fillRegion(long startFlatIndex, long endFlatIndex, byte value);

    default void fill(byte value) {
        fillRegion(0, getNumberOfElements(), value);
    }

    void fillRegion(long startFlatIndex, long endFlatIndex, short value);

    default void fill(short value) {
        fillRegion(0, getNumberOfElements(), value);
    }

    void fillRegion(long startFlatIndex, long endFlatIndex, int value);

    default void fill(int value) {
        fillRegion(0, getNumberOfElements(), value);
    }

    void fillRegion(long startFlatIndex, long endFlatIndex, long value);

    default void fill(long value) {
        fillRegion(0, getNumberOfElements(), value);
    }

    void fillRegion(long startFlatIndex, long endFlatIndex, float value);

    default void fill(float value) {
        fillRegion(0, getNumberOfElements(), value);
    }

    void fillRegion(long startFlatIndex, long endFlatIndex, double value);

    default void fill(double value) {
        fillRegion(0, getNumberOfElements(), value);
    }

    void fillRegion(long startFlatIndex, long endFlatIndex, boolean value);

    default void fill(boolean value) {
        fillRegion(0, getNumberOfElements(), value);
    }

    @NotNull ITensor where(byte condition, byte x, byte y);

    @NotNull ITensor where(short condition, short x, short y);

    @NotNull ITensor where(int condition, int x, int y);

    @NotNull ITensor where(long condition, long x, long y);

    @NotNull ITensor where(float condition, float x, float y);

    @NotNull ITensor where(double condition, double x, double y);

    @NotNull ITensor exp();

    @NotNull ITensor softmax(int dimension);

    @NotNull ITensor reduceSum(int dimension);

    @NotNull ITensor reduceSum(int dimension, boolean keepDimensions);

    @NotNull
    default ITensor reduceSum(@NotNull List<Integer> dimensions, boolean keepDimensions) {
        int deletedDimensions = 0;
        ITensor result = this;
        for (Integer dimension : dimensions) {
            result = result.reduceSum(dimension - deletedDimensions, keepDimensions);
            if (keepDimensions) {
                deletedDimensions++;
            }
        }
        return result;
    }

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

    @NotNull ITensor tanh();

    @NotNull ITensor log();

    @NotNull ITensor argmax(int dimension);

    /**
     * Compares the elements of this tensor with the elements of the other tensor.
     * The tensors must have the same shape.
     *
     * @param other the other tensor.
     * @return the result of the comparison.
     */
    @NotNull ITensor compareElements(@NotNull ITensor other);

    /**
     * Takes a tensor with index values of shape (*) and returns a tensor (*, numClasses) that have zeros everywhere except
     * at the index specified by the index tensor, where the value is 1.
     * The data type of the original tensor must be an integer type.
     * The data type of the original tensor is preserved in the result.
     *
     * @param numClasses the number of classes.
     * @return the result of the operation.
     */
    @NotNull ITensor oneHot(long numClasses);

    @NotNull ITensor get(@NotNull ITensor... indicesTensors);

    @NotNull
    default ITensor mean(int dimension) {
        try (ITensor sum = reduceSum(dimension, false)) {
            return sum.divide(getNumberOfElements());
        }
    }

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

    default <T extends ITensor> @Nullable T getIfIsType(@NotNull Class<T> typeClass) {
        return typeClass.isInstance(this) ? typeClass.cast(this) : null;
    }

    /**
     * Disposes the resources of the tensor. Note that it is optional to call this method, as the tensor's resources will be disposed when the tensor is garbage collected.
     * This method is implemented to provide a hint to the GC that the resources may be garbage collected.
     */
    @Override
    void dispose();

    /**
     * @return true, if the tensor is disposed.
     */
    boolean isDisposed();

    @NotNull ITensor cast(@NotNull DataType dataType);

    /**
     * @return the number of bytes used by the tensor's data.
     */
    default long getNumBytes() {
        return getDataType().getSizeOf(getNumberOfElements());
    }

    /**
     * Close method invoked on the tensor will mark it "de-referenced" and will be immediately disposed, once the recording scope is closed.
     */
    @Override
    void close();

    /**
     * @return true if the tensor was closed
     */
    boolean isDeReferenced();


    /**
     * Makes the tensor reference its associated graph node to prevent it from being garbage collected before this tensor is.
     *
     * @param graphNode the graph node
     */
    void setAssociatedGraphNode(@Nullable IGraph.ITensorNode graphNode);

    @Nullable
    IGraph.ITensorNode getAssociatedGraphNode();

    /**
     * Writes the contents in tightly packed binary format to the specified output stream.
     * Uses BIG_ENDIAN byte order.
     *
     * @param outputStream the output stream to write to.
     * @throws IOException if an I/O error occurs.
     */
    void writeTo(@NotNull OutputStream outputStream) throws IOException;

    /**
     * Reads the contents in tightly packed binary format from the specified input stream and sets them as contents of this tensor.
     *
     * @param inputStream the input stream to read from.
     * @throws IOException if an I/O error occurs.
     */
    void readFrom(@NotNull InputStream inputStream) throws IOException;
}