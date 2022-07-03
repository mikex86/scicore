package me.mikex86.scicore;

import me.mikex86.scicore.backend.SciCoreBackend;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

class View implements ITensor {

    private final @NotNull ITensor viewed;
    private final long @NotNull [] shape;
    private final long offset;
    private final long[] localStrides;

    View(@NotNull ITensor viewed, long @NotNull [] shape, long offset, long[] localStrides) {
        this.viewed = viewed;
        this.shape = shape;
        this.offset = offset;
        this.localStrides = localStrides;
    }

    @Override
    public @NotNull DataType getDataType() {
        return this.viewed.getDataType();
    }

    @Override
    public long @NotNull [] getShape() {
        return this.shape;
    }

    @Override
    public byte getByte(long @NotNull ... indices) {
        validateDataType(DataType.INT8);
        validateIndices(indices);
        long flatIndex = ShapeUtils.getFlatIndex(indices, this.localStrides);
        return this.viewed.getByteFlat(this.offset + flatIndex);
    }

    @Override
    public short getShort(long @NotNull ... indices) {
        validateDataType(DataType.INT16);
        validateIndices(indices);
        long flatIndex = ShapeUtils.getFlatIndex(indices, this.localStrides);
        return this.viewed.getShortFlat(this.offset + flatIndex);
    }

    @Override
    public int getInt(long @NotNull ... indices) {
        validateDataType(DataType.INT32);
        validateIndices(indices);
        long flatIndex = ShapeUtils.getFlatIndex(indices, this.localStrides);
        return this.viewed.getIntFlat(this.offset + flatIndex);
    }

    @Override
    public long getLong(long @NotNull ... indices) {
        validateDataType(DataType.INT64);
        validateIndices(indices);
        long flatIndex = ShapeUtils.getFlatIndex(indices, this.localStrides);
        return this.viewed.getLongFlat(this.offset + flatIndex);
    }

    @Override
    public float getFloat(long @NotNull ... indices) {
        validateDataType(DataType.FLOAT32);
        validateIndices(indices);
        long flatIndex = ShapeUtils.getFlatIndex(indices, this.localStrides);
        return this.viewed.getFloatFlat(this.offset + flatIndex);
    }

    @Override
    public double getDouble(long @NotNull ... indices) {
        validateDataType(DataType.FLOAT64);
        validateIndices(indices);
        long flatIndex = ShapeUtils.getFlatIndex(indices, this.localStrides);
        return this.viewed.getDoubleFlat(this.offset + flatIndex);
    }

    @Override
    public void setByte(byte value, long @NotNull ... indices) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setShort(short value, long @NotNull ... indices) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setInt(int value, long @NotNull ... indices) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setLong(long value, long @NotNull ... indices) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setFloat(float value, long @NotNull ... indices) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setDouble(double value, long @NotNull ... indices) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public byte getByteFlat(long flatIndex) {
        return this.viewed.getByteFlat(this.offset + flatIndex);
    }

    @Override
    public short getShortFlat(long flatIndex) {
        return this.viewed.getShortFlat(this.offset + flatIndex);
    }

    @Override
    public int getIntFlat(long flatIndex) {
        return this.viewed.getIntFlat(this.offset + flatIndex);
    }

    @Override
    public long getLongFlat(long flatIndex) {
        return this.viewed.getLongFlat(this.offset + flatIndex);
    }

    @Override
    public float getFloatFlat(long flatIndex) {
        return this.viewed.getFloatFlat(this.offset + flatIndex);
    }

    @Override
    public double getDoubleFlat(long flatIndex) {
        return this.viewed.getDoubleFlat(this.offset + flatIndex);
    }

    @Override
    public void setByteFlat(byte value, long flatIndex) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setShortFlat(short value, long flatIndex) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setIntFlat(int value, long flatIndex) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setLongFlat(long value, long flatIndex) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setFloatFlat(float value, long flatIndex) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setDoubleFlat(double value, long flatIndex) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public @NotNull ITensor copy() {
        return this;
    }

    @Override
    public void setContents(@NotNull ITensor tensor) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setContents(long @NotNull [] dimension, @NotNull ITensor tensor, boolean useView) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public @NotNull ITensor multiplied(@NotNull Scalar s) {
        return ITensor.super.multiplied(s);
    }

    @Override
    public void fill(byte i) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void fill(short i) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void fill(int i) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void fill(long i) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void fill(float i) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void fill(double i) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public @NotNull ITensor exp() {
        return ITensor.super.exp();
    }

    @Override
    public @NotNull ITensorIterator iterator() {
        // TODO: IMPLEMENT
        throw new UnsupportedOperationException("NOT YET IMPLEMENTED");
    }

    @Override
    public @NotNull SciCoreBackend getSciCore() {
        return this.viewed.getSciCore();
    }

}
