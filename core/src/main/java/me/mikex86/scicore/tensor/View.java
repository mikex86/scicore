package me.mikex86.scicore.tensor;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.nio.*;

public class View extends AbstractTensor {

    private final @NotNull ITensor viewed;
    private final long @NotNull [] shape;
    private final long offset;
    private final long[] localStrides;

    View(@NotNull ITensor viewed, long @NotNull [] shape, long offset, long[] localStrides) {
        this.numElements = ShapeUtils.getNumElements(shape);
        this.viewed = viewed;
        this.shape = shape;
        this.offset = offset;
        this.localStrides = localStrides;
    }

    public long getOffset() {
        return offset;
    }

    @NotNull
    public ITensor getViewed() {
        return viewed;
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
    public long @NotNull [] getStrides() {
        return this.localStrides;
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
    public boolean getBooleanFlat(long flatIndex) {
        validateDataType(DataType.BOOLEAN);
        return this.viewed.getBooleanFlat(this.offset + flatIndex);
    }

    @Override
    public void setBooleanFlat(boolean value, long flatIndex) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public boolean getBoolean(long @NotNull ... indices) {
        validateDataType(DataType.BOOLEAN);
        validateIndices(indices);
        long flatIndex = ShapeUtils.getFlatIndex(indices, this.localStrides);
        return this.viewed.getBooleanFlat(this.offset + flatIndex);
    }

    @Override
    public void setBoolean(boolean value, long @NotNull ... indices) {
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
    public void setContents(@NotNull ByteBuffer buffer) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setContents(@NotNull ShortBuffer buffer) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setContents(@NotNull IntBuffer buffer) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setContents(@NotNull LongBuffer buffer) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setContents(@NotNull FloatBuffer buffer) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setContents(@NotNull DoubleBuffer buffer) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setContents(boolean @NotNull [] buffer) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setContents(long @NotNull [] index, @NotNull ITensor tensor) {
        throw new UnsupportedOperationException("Views are read-only");
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
    public void fill(boolean value) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public @NotNull ISciCoreBackend getSciCoreBackend() {
        return this.viewed.getSciCoreBackend();
    }

    @Override
    public @NotNull DirectMemoryHandle getContentsAsDirectMemory() {
        long nElements = getNumberOfElements();
        return this.viewed.getContentsAsDirectMemory(this.offset, this.offset + nElements);
    }

    @Override
    public @NotNull DirectMemoryHandle getContentsAsDirectMemory(long startFlatIndex, long endFlatIndex) {
        if (startFlatIndex < 0 || endFlatIndex > getNumberOfElements()) {
            throw new IllegalArgumentException("Invalid flat indices");
        }
        return this.viewed.getContentsAsDirectMemory(this.offset + startFlatIndex, this.offset + endFlatIndex);
    }
}