package me.mikex86.scicore.backend;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.ITensorIterator;
import me.mikex86.scicore.backend.impl.jvm.JvmScalarImpl;
import me.mikex86.scicore.backend.impl.jvm.JvmTensorImpl;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

public interface TensorImpl {

    @NotNull
    DataType getDataType();

    long @NotNull [] getShape();

    long @NotNull [] getStrides();

    byte getByte(long @NotNull [] indices);

    short getShort(long @NotNull [] indices);

    int getInt(long @NotNull [] indices);

    long getLong(long @NotNull [] indices);

    float getFloat(long @NotNull [] indices);

    double getDouble(long @NotNull [] indices);

    byte getByteFlat(long flatIndex);

    short getShortFlat(long flatIndex);

    int getIntFlat(long flatIndex);

    long getLongFlat(long flatIndex);

    float getFloatFlat(long flatIndex);

    double getDoubleFlat(long flatIndex);

    void setByte(byte value, long @NotNull [] indices);

    void setShort(short value, long @NotNull [] indices);

    void setInt(int value, long @NotNull [] indices);

    void setLong(long value, long @NotNull [] indices);

    void setFloat(float value, long @NotNull [] indices);

    void setDouble(double value, long @NotNull [] indices);

    void setByteFlat(byte value, long flatIndex);

    void setShortFlat(short value, long flatIndex);

    void setIntFlat(int value, long flatIndex);

    void setLongFlat(long value, long flatIndex);

    void setFloatFlat(float value, long flatIndex);

    void setDoubleFlat(double value, long flatIndex);

    void setContents(@NotNull ITensor tensor);

    void setContents(long @NotNull [] dimension, @NotNull ITensor tensor, boolean useView);

    @NotNull
    JvmTensorImpl multiplied(@NotNull JvmScalarImpl b);

    void fill(byte i);

    void fill(short i);

    void fill(int i);

    void fill(long i);

    void fill(float i);

    void fill(double i);

    @Override
    String toString();

    @NotNull
    ITensorIterator iterator();

    @NotNull
    TensorImpl matmul(@NotNull JvmTensorImpl b);

    @NotNull
    default TensorImpl exp() {
        long[] shape = getShape();
        long nElements = ShapeUtils.getNumElements(shape);
        DataType dataType = getDataType();
        TensorImpl result = new JvmTensorImpl(dataType, shape);
        for (long i = 0; i < nElements; i++) {
            if (dataType.isFloatingPoint()) {
                double value = switch (dataType) {
                    case FLOAT32 -> getFloatFlat(i);
                    case FLOAT64 -> getDoubleFlat(i);
                    default -> throw new IllegalStateException("Unexpected value: " + dataType);
                };
                switch (dataType) {
                    case FLOAT32 -> result.setFloatFlat((float) Math.exp(value), i);
                    case FLOAT64 -> result.setDoubleFlat(Math.exp(value), i);
                }
            } else {
                long value = switch (dataType) {
                    case INT8 -> getByteFlat(i);
                    case INT16 -> getShortFlat(i);
                    case INT32 -> getIntFlat(i);
                    case INT64 -> getLongFlat(i);
                    default -> throw new IllegalStateException("Unexpected value: " + dataType);
                };
                switch (dataType) {
                    case INT8 -> result.setByteFlat((byte) Math.exp(value), i);
                    case INT16 -> result.setShortFlat((short) Math.exp(value), i);
                    case INT32 -> result.setIntFlat((int) Math.exp(value), i);
                    case INT64 -> result.setLongFlat((long) Math.exp(value), i);
                }
            }
        }
        return result;
    }
}
