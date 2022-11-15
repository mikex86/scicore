package me.mikex86.scicore.tensor;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.io.InputStream;
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
    public boolean getBooleanFlat(long flatIndex) {
        validateDataType(DataType.BOOLEAN);
        return this.viewed.getBooleanFlat(this.offset + flatIndex);
    }

    @Override
    public void setBooleanFlat(boolean value, long flatIndex) {
        validateDataType(DataType.BOOLEAN);
        this.viewed.setBooleanFlat(value, this.offset + flatIndex);
    }

    @Override
    public byte getByteFlat(long flatIndex) {
        validateDataType(DataType.INT8);
        return this.viewed.getByteFlat(this.offset + flatIndex);
    }

    @Override
    public short getShortFlat(long flatIndex) {
        validateDataType(DataType.INT16);
        return this.viewed.getShortFlat(this.offset + flatIndex);
    }

    @Override
    public int getIntFlat(long flatIndex) {
        validateDataType(DataType.INT32);
        return this.viewed.getIntFlat(this.offset + flatIndex);
    }

    @Override
    public long getLongFlat(long flatIndex) {
        validateDataType(DataType.INT64);
        return this.viewed.getLongFlat(this.offset + flatIndex);
    }

    @Override
    public float getFloatFlat(long flatIndex) {
        validateDataType(DataType.FLOAT32);
        return this.viewed.getFloatFlat(this.offset + flatIndex);
    }

    @Override
    public double getDoubleFlat(long flatIndex) {
        validateDataType(DataType.FLOAT64);
        return this.viewed.getDoubleFlat(this.offset + flatIndex);
    }

    @Override
    public void setByteFlat(byte value, long flatIndex) {
        validateDataType(DataType.INT8);
        this.viewed.setByteFlat(value, this.offset + flatIndex);
    }

    @Override
    public void setShortFlat(short value, long flatIndex) {
        validateDataType(DataType.INT16);
        this.viewed.setShortFlat(value, this.offset + flatIndex);
    }

    @Override
    public void setIntFlat(int value, long flatIndex) {
        validateDataType(DataType.INT32);
        this.viewed.setIntFlat(value, this.offset + flatIndex);
    }

    @Override
    public void setLongFlat(long value, long flatIndex) {
        validateDataType(DataType.INT64);
        this.viewed.setLongFlat(value, this.offset + flatIndex);
    }

    @Override
    public void setFloatFlat(float value, long flatIndex) {
        validateDataType(DataType.FLOAT32);
        this.viewed.setFloatFlat(value, this.offset + flatIndex);
    }

    @Override
    public void setDoubleFlat(double value, long flatIndex) {
        validateDataType(DataType.FLOAT64);
        this.viewed.setDoubleFlat(value, this.offset + flatIndex);
    }

    @Override
    public @NotNull ITensor copy() {
        ISciCoreBackend backend = this.viewed.getSciCoreBackend();
        ITensor copy = backend.createTensor(this.viewed.getDataType(), this.shape);
        copy.setContents(this);
        return copy;
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull ITensor tensor) {
        validateDataType(tensor.getDataType());
        this.viewed.setContentsWithOffset(this.offset + startFlatIndex, tensor);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull ByteBuffer buffer) {
        this.viewed.setContentsWithOffset(this.offset + startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull ShortBuffer buffer) {
        validateDataType(DataType.INT16);
        this.viewed.setContentsWithOffset(this.offset + startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull IntBuffer buffer) {
        validateDataType(DataType.INT32);
        this.viewed.setContentsWithOffset(this.offset + startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull LongBuffer buffer) {
        validateDataType(DataType.INT64);
        this.viewed.setContentsWithOffset(this.offset + startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull FloatBuffer buffer) {
        validateDataType(DataType.FLOAT32);
        this.viewed.setContentsWithOffset(this.offset + startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull DoubleBuffer buffer) {
        validateDataType(DataType.FLOAT64);
        this.viewed.setContentsWithOffset(this.offset + startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, boolean @NotNull [] buffer) {
        validateDataType(DataType.BOOLEAN);
        this.viewed.setContentsWithOffset(this.offset + startFlatIndex, buffer);
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

    @Override
    public void readFrom(@NotNull InputStream inputStream) throws IOException {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void dispose() {
        // Do nothing, not even call super.dispose() which would mark the tensor as disposed, which we don't want
    }
}
