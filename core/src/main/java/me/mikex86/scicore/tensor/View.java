package me.mikex86.scicore.tensor;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.tensor.data.ITensorDataContainer;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.io.InputStream;
import java.nio.*;
import java.util.Arrays;

public class View extends AbstractTensor {

    private final @NotNull ITensorDataContainer viewed;
    private final long @NotNull [] shape;
    private final long offset;
    private final long[] localStrides;

    @NotNull
    private final ISciCoreBackend backend;

    public View(@NotNull ISciCoreBackend backend, @NotNull ITensorDataContainer viewed, long @NotNull [] shape, long offset, long[] localStrides) {
        this.backend = backend;
        this.numElements = ShapeUtils.getNumElements(shape);
        this.viewed = viewed;
        this.shape = shape;
        this.offset = offset;
        this.localStrides = localStrides;
    }

    @Override
    @NotNull
    public ITensor getView(long @NotNull ... indices) {
        long[] shape = getShape();
        validateIndices(indices);
        long[] strides = getStrides();

        long[] sliceShape = Arrays.copyOfRange(shape, indices.length, shape.length);
        long[] sliceStrides = ShapeUtils.makeStrides(sliceShape);

        long offset = this.offset + ShapeUtils.getFlatIndex(indices, shape, strides);
        return new View(getSciCoreBackend(), getDataContainer(), sliceShape, offset, sliceStrides);
    }

    public long getOffset() {
        return offset;
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
    public boolean getBooleanFlat(long flatIndex) {
        validateDataType(DataType.BOOLEAN);
        return this.viewed.getBooleanFlat(this.offset + flatIndex);
    }

    @Override
    public void setBooleanFlat(boolean value, long flatIndex) {
        validateDataType(DataType.BOOLEAN);
        this.viewed.setBooleanFlat(this.offset + flatIndex, value);
    }

    @Override
    public byte getByteFlat(long flatIndex) {
        validateDataType(DataType.INT8);
        return this.viewed.getInt8Flat(this.offset + flatIndex);
    }

    @Override
    public short getShortFlat(long flatIndex) {
        validateDataType(DataType.INT16);
        return this.viewed.getInt16Flat(this.offset + flatIndex);
    }

    @Override
    public int getIntFlat(long flatIndex) {
        validateDataType(DataType.INT32);
        return this.viewed.getInt32Flat(this.offset + flatIndex);
    }

    @Override
    public long getLongFlat(long flatIndex) {
        validateDataType(DataType.INT64);
        return this.viewed.getInt64Flat(this.offset + flatIndex);
    }

    @Override
    public float getFloatFlat(long flatIndex) {
        validateDataType(DataType.FLOAT32);
        return this.viewed.getFloat32Flat(this.offset + flatIndex);
    }

    @Override
    public double getDoubleFlat(long flatIndex) {
        validateDataType(DataType.FLOAT64);
        return this.viewed.getFloat64Flat(this.offset + flatIndex);
    }

    @Override
    public void setByteFlat(byte value, long flatIndex) {
        validateDataType(DataType.INT8);
        this.viewed.setInt8Flat(this.offset + flatIndex, value);
    }

    @Override
    public void setShortFlat(short value, long flatIndex) {
        validateDataType(DataType.INT16);
        this.viewed.setInt16Flat(this.offset + flatIndex, value);
    }

    @Override
    public void setIntFlat(int value, long flatIndex) {
        validateDataType(DataType.INT32);
        this.viewed.setInt32Flat(this.offset + flatIndex, value);
    }

    @Override
    public void setLongFlat(long value, long flatIndex) {
        validateDataType(DataType.INT64);
        this.viewed.setInt64Flat(this.offset + flatIndex, value);
    }

    @Override
    public void setFloatFlat(float value, long flatIndex) {
        validateDataType(DataType.FLOAT32);
        this.viewed.setFloat32Flat(this.offset + flatIndex, value);
    }

    @Override
    public void setDoubleFlat(double value, long flatIndex) {
        validateDataType(DataType.FLOAT64);
        this.viewed.setFloat64Flat(this.offset + flatIndex, value);
    }

    @Override
    public @NotNull ITensor copy() {
        ITensor copy = backend.createTensor(this.viewed.getDataType(), this.shape);
        copy.setContents(this);
        return copy;
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull ITensor tensor) {
        validateDataType(tensor.getDataType());
        if (tensor.getDataType() != getDataType()) {
            throw new IllegalArgumentException("Cannot copy tensor with different data type");
        }
        long numElements = tensor.getNumberOfElements();
        if (numElements > this.numElements - startFlatIndex) {
            throw new IllegalArgumentException("Cannot copy tensor with more elements than the destination tensor can hold");
        }
        DirectMemoryHandle directMemory = tensor.getContentsAsDirectMemory();
        this.viewed.setContents(this.offset + startFlatIndex, directMemory.asByteBuffer());
        if (directMemory.canFree()) {
            directMemory.free();
        }
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull ByteBuffer buffer) {
        this.viewed.setContents(this.offset + startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull ShortBuffer buffer) {
        validateDataType(DataType.INT16);
        this.viewed.setContents(this.offset + startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull IntBuffer buffer) {
        validateDataType(DataType.INT32);
        this.viewed.setContents(this.offset + startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull LongBuffer buffer) {
        validateDataType(DataType.INT64);
        this.viewed.setContents(this.offset + startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull FloatBuffer buffer) {
        validateDataType(DataType.FLOAT32);
        this.viewed.setContents(this.offset + startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull DoubleBuffer buffer) {
        validateDataType(DataType.FLOAT64);
        this.viewed.setContents(this.offset + startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, boolean @NotNull [] buffer) {
        validateDataType(DataType.BOOLEAN);
        this.viewed.setContents(this.offset + startFlatIndex, buffer);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, byte value) {
        this.viewed.fillRegion(this.offset + startFlatIndex, this.offset + endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, short value) {
        validateDataType(DataType.INT16);
        this.viewed.fillRegion(this.offset + startFlatIndex, this.offset + endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, int value) {
        validateDataType(DataType.INT32);
        this.viewed.fillRegion(this.offset + startFlatIndex, this.offset + endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, long value) {
        validateDataType(DataType.INT64);
        this.viewed.fillRegion(this.offset + startFlatIndex, this.offset + endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, float value) {
        validateDataType(DataType.FLOAT32);
        this.viewed.fillRegion(this.offset + startFlatIndex, this.offset + endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, double value) {
        validateDataType(DataType.FLOAT64);
        this.viewed.fillRegion(this.offset + startFlatIndex, this.offset + endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, boolean value) {
        validateDataType(DataType.BOOLEAN);
        this.viewed.fillRegion(this.offset + startFlatIndex, this.offset + endFlatIndex, value);
    }


    @Override
    public @NotNull ISciCoreBackend getSciCoreBackend() {
        return backend;
    }

    @Override
    public @NotNull DirectMemoryHandle getContentsAsDirectMemory() {
        long nElements = getNumberOfElements();
        return this.viewed.getAsDirectBuffer(this.offset, this.offset + nElements);
    }

    @Override
    public @NotNull DirectMemoryHandle getContentsAsDirectMemory(long startFlatIndex, long endFlatIndex) {
        if (startFlatIndex < 0 || endFlatIndex > getNumberOfElements()) {
            throw new IllegalArgumentException("Invalid flat indices");
        }
        return this.viewed.getAsDirectBuffer(this.offset + startFlatIndex, this.offset + endFlatIndex);
    }

    @Override
    public void readFrom(@NotNull InputStream inputStream) throws IOException {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void dispose() {
        // Do nothing, not even call super.dispose() which would mark the tensor as disposed, which we don't want
    }

    @Override
    public @NotNull ITensorDataContainer getDataContainer() {
        return viewed;
    }
}
