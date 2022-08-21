package me.mikex86.scicore.backend.impl.jvm;

import me.mikex86.scicore.*;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.op.IDerivedTensor;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;
import java.util.function.Supplier;

public class JvmDerivedTensor extends AbstractTensor implements IDerivedTensor {

    private final long @NotNull [] resultShape;

    @NotNull
    private final DataType resultDataType;

    @NotNull
    private final Supplier<ITensor> resultSupplier;

    @Nullable
    private ITensor lazyResult;

    @NotNull
    private final ISciCoreBackend sciCoreBackend;

    public JvmDerivedTensor(@NotNull ISciCoreBackend backend, long @NotNull [] resultShape, @NotNull DataType resultDataType, @NotNull Supplier<ITensor> resultSupplier) {
        this.resultShape = resultShape;
        this.resultDataType = resultDataType;
        this.resultSupplier = resultSupplier;
        this.lazyResult = resultSupplier.get();
        this.sciCoreBackend = backend;
    }

    @NotNull
    private ITensor result() {
        if (lazyResult == null) {
            lazyResult = resultSupplier.get();
        }
        return lazyResult;
    }

    @Override
    public @NotNull DataType getDataType() {
        return resultDataType;
    }

    @Override
    public long @NotNull [] getShape() {
        return resultShape;
    }

    @Override
    public long @NotNull [] getStrides() {
        return result().getStrides();
    }

    @Override
    public byte getByte(long @NotNull ... indices) {
        return result().getByte(indices);
    }

    @Override
    public short getShort(long @NotNull ... indices) {
        return result().getShort(indices);
    }

    @Override
    public int getInt(long @NotNull ... indices) {
        return result().getInt(indices);
    }

    @Override
    public long getLong(long @NotNull ... indices) {
        return result().getLong(indices);
    }

    @Override
    public float getFloat(long @NotNull ... indices) {
        return result().getFloat(indices);
    }

    @Override
    public double getDouble(long @NotNull ... indices) {
        return result().getDouble(indices);
    }

    @Override
    public void setByte(byte value, long @NotNull ... indices) {
        result().setByte(value, indices);
    }

    @Override
    public void setShort(short value, long @NotNull ... indices) {
        result().setShort(value, indices);
    }

    @Override
    public void setInt(int value, long @NotNull ... indices) {
        result().setInt(value, indices);
    }

    @Override
    public void setLong(long value, long @NotNull ... indices) {
        result().setLong(value, indices);
    }

    @Override
    public void setFloat(float value, long @NotNull ... indices) {
        result().setFloat(value, indices);
    }

    @Override
    public void setDouble(double value, long @NotNull ... indices) {
        result().setDouble(value, indices);
    }

    @Override
    public boolean getBooleanFlat(long flatIndex) {
        return result().getBooleanFlat(flatIndex);
    }

    @Override
    public void setBooleanFlat(boolean value, long flatIndex) {
        result().setBooleanFlat(value, flatIndex);
    }

    @Override
    public boolean getBoolean(long @NotNull ... indices) {
        return result().getBoolean(indices);
    }

    @Override
    public void setBoolean(boolean value, long @NotNull ... indices) {
        result().setBoolean(value, indices);
    }

    @Override
    public byte getByteFlat(long flatIndex) {
        return result().getByteFlat(flatIndex);
    }

    @Override
    public short getShortFlat(long flatIndex) {
        return result().getShortFlat(flatIndex);
    }

    @Override
    public int getIntFlat(long flatIndex) {
        return result().getIntFlat(flatIndex);
    }

    @Override
    public long getLongFlat(long flatIndex) {
        return result().getLongFlat(flatIndex);
    }

    @Override
    public float getFloatFlat(long flatIndex) {
        return result().getFloatFlat(flatIndex);
    }

    @Override
    public double getDoubleFlat(long flatIndex) {
        return result().getDoubleFlat(flatIndex);
    }

    @Override
    public void setByteFlat(byte value, long flatIndex) {
        result().setByteFlat(value, flatIndex);
    }

    @Override
    public void setShortFlat(short value, long flatIndex) {
        result().setShortFlat(value, flatIndex);
    }

    @Override
    public void setIntFlat(int value, long flatIndex) {
        result().setIntFlat(value, flatIndex);
    }

    @Override
    public void setLongFlat(long value, long flatIndex) {
        result().setLongFlat(value, flatIndex);
    }

    @Override
    public void setFloatFlat(float value, long flatIndex) {
        result().setFloatFlat(value, flatIndex);
    }

    @Override
    public void setDoubleFlat(double value, long flatIndex) {
        result().setDoubleFlat(value, flatIndex);
    }

    @Override
    public @NotNull ITensor copy() {
        return result().copy();
    }

    @Override
    public void setContents(@NotNull ITensor tensor) {
        result().setContents(tensor);
    }

    @Override
    public void setContents(long @NotNull [] dimension, @NotNull ITensor tensor, boolean useView) {
        result().setContents(dimension, tensor, useView);
    }

    @Override
    public void fill(byte i) {
        result().fill(i);
    }

    @Override
    public void fill(short i) {
        result().fill(i);
    }

    @Override
    public void fill(int i) {
        result().fill(i);
    }

    @Override
    public void fill(long i) {
        result().fill(i);
    }

    @Override
    public void fill(float i) {
        result().fill(i);
    }

    @Override
    public void fill(double i) {
        result().fill(i);
    }

    @Override
    public void fill(boolean value) {
        result().fill(value);
    }

    @Override
    public @NotNull ITensor exp() {
        return result().exp();
    }

    @Override
    public @NotNull ITensorIterator iterator() {
        return result().iterator();
    }

    @Override
    public @NotNull ISciCoreBackend getSciCoreBackend() {
        return this.sciCoreBackend;
    }

    @Override
    public String toString() {
        return result().toString();
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof ITensor tensor)) {
            return false;
        }
        if (!Arrays.equals(tensor.getShape(), this.getShape())) {
            return false;
        }
        if (tensor.getDataType() != this.getDataType()) {
            return false;
        }
        return result().equals(obj);
    }
}
