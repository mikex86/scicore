package me.mikex86.scicore;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.backend.ITensorImpl;
import me.mikex86.scicore.backend.impl.jvm.JvmDataTensorImpl;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;

public class Tensor implements ITensor {

    @NotNull
    private final ITensorImpl tensorImpl;

    @NotNull
    private final ISciCoreBackend backend;

    Tensor(@NotNull ISciCoreBackend backend, @NotNull DataType dataType, long @NotNull [] shape) {
        this.backend = backend;
        this.tensorImpl = backend.createTensor(dataType, shape);
    }

    public Tensor(@NotNull ISciCoreBackend backend, @NotNull ITensorImpl tensorImpl) {
        this.tensorImpl = tensorImpl;
        this.backend = backend;
    }

    @Override
    public @NotNull DataType getDataType() {
        return this.tensorImpl.getDataType();
    }

    @Override
    public long @NotNull [] getShape() {
        return this.tensorImpl.getShape();
    }

    @Override
    public long @NotNull [] getStrides() {
        return this.tensorImpl.getStrides();
    }

    @Override
    public byte getByte(long @NotNull ... indices) {
        validateDataType(DataType.INT8);
        validateIndices(indices);
        return this.tensorImpl.getByte(indices);
    }

    @Override
    public short getShort(long @NotNull ... indices) {
        validateDataType(DataType.INT16);
        validateIndices(indices);
        return this.tensorImpl.getShort(indices);
    }

    @Override
    public int getInt(long @NotNull ... indices) {
        validateDataType(DataType.INT32);
        validateIndices(indices);
        return this.tensorImpl.getInt(indices);
    }

    @Override
    public long getLong(long @NotNull ... indices) {
        validateDataType(DataType.INT64);
        validateIndices(indices);
        return this.tensorImpl.getLong(indices);
    }

    @Override
    public float getFloat(long @NotNull ... indices) {
        validateDataType(DataType.FLOAT32);
        validateIndices(indices);
        return this.tensorImpl.getFloat(indices);
    }

    @Override
    public double getDouble(long @NotNull ... indices) {
        validateDataType(DataType.FLOAT64);
        validateIndices(indices);
        return this.tensorImpl.getDouble(indices);
    }

    @Override
    public void setByte(byte value, long @NotNull ... indices) {
        validateDataType(DataType.INT8);
        validateIndices(indices);
        this.tensorImpl.setByte(value, indices);
    }

    @Override
    public void setShort(short value, long @NotNull ... indices) {
        validateDataType(DataType.INT16);
        validateIndices(indices);
        this.tensorImpl.setShort(value, indices);
    }

    @Override
    public void setInt(int value, long @NotNull ... indices) {
        validateDataType(DataType.INT32);
        validateIndices(indices);
        this.tensorImpl.setInt(value, indices);
    }

    @Override
    public void setLong(long value, long @NotNull ... indices) {
        validateDataType(DataType.INT64);
        validateIndices(indices);
        this.tensorImpl.setLong(value, indices);
    }

    @Override
    public void setFloat(float value, long @NotNull ... indices) {
        validateDataType(DataType.FLOAT32);
        validateIndices(indices);
        this.tensorImpl.setFloat(value, indices);
    }

    @Override
    public void setDouble(double value, long @NotNull ... indices) {
        validateDataType(DataType.FLOAT64);
        validateIndices(indices);
        this.tensorImpl.setDouble(value, indices);
    }

    @Override
    public byte getByteFlat(long flatIndex) {
        validateDataType(DataType.INT8);
        return this.tensorImpl.getByteFlat(flatIndex);
    }

    @Override
    public short getShortFlat(long flatIndex) {
        validateDataType(DataType.INT16);
        return this.tensorImpl.getShortFlat(flatIndex);
    }

    @Override
    public int getIntFlat(long flatIndex) {
        validateDataType(DataType.INT32);
        return this.tensorImpl.getIntFlat(flatIndex);
    }

    @Override
    public long getLongFlat(long flatIndex) {
        validateDataType(DataType.INT64);
        return this.tensorImpl.getLongFlat(flatIndex);
    }

    @Override
    public float getFloatFlat(long flatIndex) {
        validateDataType(DataType.FLOAT32);
        return this.tensorImpl.getFloatFlat(flatIndex);
    }

    @Override
    public double getDoubleFlat(long flatIndex) {
        validateDataType(DataType.FLOAT64);
        return this.tensorImpl.getDoubleFlat(flatIndex);
    }

    @Override
    public void setByteFlat(byte value, long flatIndex) {
        validateDataType(DataType.INT8);
        this.tensorImpl.setByteFlat(value, flatIndex);
    }

    @Override
    public void setShortFlat(short value, long flatIndex) {
        validateDataType(DataType.INT16);
        this.tensorImpl.setShortFlat(value, flatIndex);
    }

    @Override
    public void setIntFlat(int value, long flatIndex) {
        validateDataType(DataType.INT32);
        this.tensorImpl.setIntFlat(value, flatIndex);
    }

    @Override
    public void setLongFlat(long value, long flatIndex) {
        validateDataType(DataType.INT64);
        this.tensorImpl.setLongFlat(value, flatIndex);
    }

    @Override
    public void setFloatFlat(float value, long flatIndex) {
        validateDataType(DataType.FLOAT32);
        this.tensorImpl.setFloatFlat(value, flatIndex);
    }

    @Override
    public void setDoubleFlat(double value, long flatIndex) {
        validateDataType(DataType.FLOAT64);
        this.tensorImpl.setDoubleFlat(value, flatIndex);
    }

    @Override
    public @NotNull ITensor copy() {
        ITensorImpl tensor = this.tensorImpl.copy();
        return new Tensor(backend, tensor);
    }

    @Override
    public void setContents(@NotNull ITensor tensor) {
        if (tensor.getDataType() != this.getDataType()) {
            throw new IllegalArgumentException("Tensor data types do not match");
        }
        if (!Arrays.equals(tensor.getShape(), this.getShape())) {
            throw new IllegalArgumentException("Tensor shapes do not match");
        }
        this.tensorImpl.setContents(tensor);
    }

    @Override
    public void setContents(long @NotNull [] dimension, @NotNull ITensor tensor, boolean useView) {
        this.tensorImpl.setContents(dimension, tensor, useView);
    }

    @Override
    public void fill(byte i) {
        this.tensorImpl.fill(i);
    }

    @Override
    public void fill(short i) {
        this.tensorImpl.fill(i);
    }

    @Override
    public void fill(int i) {
        this.tensorImpl.fill(i);
    }

    @Override
    public void fill(long i) {
        this.tensorImpl.fill(i);
    }

    @Override
    public void fill(float i) {
        this.tensorImpl.fill(i);
    }

    @Override
    public void fill(double i) {
        this.tensorImpl.fill(i);
    }

    @Override
    public @NotNull ITensorIterator iterator() {
        return this.tensorImpl.iterator();
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

    @Override
    public @NotNull ISciCoreBackend getSciCoreBackend() {
        return this.backend;
    }

    @NotNull
    public ITensorImpl getTensorImpl() {
        return tensorImpl;
    }

    @Override
    public String toString() {
        return tensorImpl.toString();
    }
}
