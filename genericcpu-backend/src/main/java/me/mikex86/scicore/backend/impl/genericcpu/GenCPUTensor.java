package me.mikex86.scicore.backend.impl.genericcpu;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.tensor.AbstractTensor;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.io.InputStream;
import java.nio.*;

public class GenCPUTensor extends AbstractTensor implements ITensor {

    @NotNull
    private final GenCPUTensorDataContainer dataContainer;

    private final long @NotNull [] strides;

    @NotNull
    private final GenCPUBackend backend;

    private final long @NotNull [] shape;

    public GenCPUTensor(@NotNull GenCPUBackend backend, @NotNull DataType dataType, long @NotNull [] shape) {
        this.numElements = ShapeUtils.getNumElements(shape);
        this.dataContainer = new GenCPUTensorDataContainer(backend.getMemoryManager(), this.numElements, dataType);
        this.dataContainer.incRc();
        this.strides = ShapeUtils.makeStrides(shape);
        this.shape = shape;
        this.backend = backend;
    }

    public GenCPUTensor(@NotNull GenCPUBackend backend, @NotNull GenCPUTensorDataContainer dataContainer, long @NotNull [] shape) {
        dataContainer.incRc();
        this.numElements = ShapeUtils.getNumElements(shape);
        this.backend = backend;
        this.dataContainer = dataContainer;
        this.shape = shape;
        this.strides = ShapeUtils.makeStrides(shape);
    }

    @Override
    public @NotNull DataType getDataType() {
        return this.dataContainer.getDataType();
    }

    @Override
    public long @NotNull [] getShape() {
        return this.shape;
    }

    @Override
    public long @NotNull [] getStrides() {
        return this.strides;
    }

    @Override
    public boolean getBooleanFlat(long flatIndex) {
        validateDataType(DataType.BOOLEAN);
        return this.dataContainer.getBooleanFlat(flatIndex);
    }

    @Override
    public void setBooleanFlat(boolean value, long flatIndex) {
        validateDataType(DataType.BOOLEAN);
        this.dataContainer.setBooleanFlat(flatIndex, value);
    }

    @Override
    public byte getByteFlat(long flatIndex) {
        validateDataType(DataType.INT8);
        return this.dataContainer.getInt8Flat(flatIndex);
    }

    @Override
    public void setByteFlat(byte value, long flatIndex) {
        validateDataType(DataType.INT8);
        this.dataContainer.setInt8Flat(flatIndex, value);
    }

    @Override
    public short getShortFlat(long flatIndex) {
        validateDataType(DataType.INT16);
        return this.dataContainer.getInt16Flat(flatIndex);
    }

    @Override
    public void setShortFlat(short value, long flatIndex) {
        validateDataType(DataType.INT16);
        this.dataContainer.setInt16Flat(flatIndex, value);
    }

    @Override
    public int getIntFlat(long flatIndex) {
        validateDataType(DataType.INT32);
        return this.dataContainer.getInt32Flat(flatIndex);
    }

    @Override
    public void setIntFlat(int value, long flatIndex) {
        validateDataType(DataType.INT32);
        this.dataContainer.setInt32Flat(flatIndex, value);
    }

    @Override
    public long getLongFlat(long flatIndex) {
        validateDataType(DataType.INT64);
        return this.dataContainer.getInt64Flat(flatIndex);
    }

    @Override
    public void setLongFlat(long value, long flatIndex) {
        validateDataType(DataType.INT64);
        this.dataContainer.setInt64Flat(flatIndex, value);
    }

    @Override
    public float getFloatFlat(long flatIndex) {
        validateDataType(DataType.FLOAT32);
        return this.dataContainer.getFloat32Flat(flatIndex);
    }

    @Override
    public void setFloatFlat(float value, long flatIndex) {
        validateDataType(DataType.FLOAT32);
        this.dataContainer.setFloat32Flat(flatIndex, value);
    }

    @Override
    public double getDoubleFlat(long flatIndex) {
        validateDataType(DataType.FLOAT64);
        return this.dataContainer.getFloat64Flat(flatIndex);
    }

    @Override
    public void setDoubleFlat(double value, long flatIndex) {
        validateDataType(DataType.FLOAT64);
        this.dataContainer.setFloat64Flat(flatIndex, value);
    }

    @Override
    public @NotNull ITensor copy() {
        ITensor copy = new GenCPUTensor(backend, getDataType(), getShape());
        copy.setContents(this);
        return copy;
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull ITensor tensor) {
        if (tensor.getDataType() != getDataType()) {
            throw new IllegalArgumentException("Cannot copy tensor with different data type");
        }
        long numElements = tensor.getNumberOfElements();
        if (numElements > this.numElements - startFlatIndex) {
            throw new IllegalArgumentException("Cannot copy tensor with more elements than the destination tensor can hold");
        }
        DirectMemoryHandle directMemory = tensor.getContentsAsDirectMemory();
        this.dataContainer.setContents(startFlatIndex, directMemory.asByteBuffer());
        if (directMemory.canFree()) {
            directMemory.free();
        }
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull ByteBuffer buffer) {
        this.dataContainer.setContents(startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull ShortBuffer buffer) {
        validateDataType(DataType.INT16);
        this.dataContainer.setContents(startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull IntBuffer buffer) {
        this.dataContainer.setContents(startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull LongBuffer buffer) {
        this.dataContainer.setContents(startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull FloatBuffer buffer) {
        this.dataContainer.setContents(startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull DoubleBuffer buffer) {
        this.dataContainer.setContents(startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, boolean @NotNull [] buffer) {
        this.dataContainer.setContents(startFlatIndex, buffer);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, byte value) {
        switch (getDataType()) {
            case INT8 -> dataContainer.fillRegion(startFlatIndex, endFlatIndex, value);
            case INT16 -> dataContainer.fillRegion(startFlatIndex, endFlatIndex, (short) value);
            case INT32 -> dataContainer.fillRegion(startFlatIndex, endFlatIndex, (int) value);
            case INT64 -> dataContainer.fillRegion(startFlatIndex, endFlatIndex, (long) value);
            case FLOAT32 -> dataContainer.fillRegion(startFlatIndex, endFlatIndex, (float) value);
            case FLOAT64 -> dataContainer.fillRegion(startFlatIndex, endFlatIndex, (double) value);
        }
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, short value) {
        switch (getDataType()) {
            case INT16 -> dataContainer.fillRegion(startFlatIndex, endFlatIndex, value);
            case INT32 -> dataContainer.fillRegion(startFlatIndex, endFlatIndex, (int) value);
            case INT64 -> dataContainer.fillRegion(startFlatIndex, endFlatIndex, (long) value);
            case FLOAT32 -> dataContainer.fillRegion(startFlatIndex, endFlatIndex, (float) value);
            case FLOAT64 -> dataContainer.fillRegion(startFlatIndex, endFlatIndex, (double) value);
            default ->
                    throw new IllegalArgumentException("Cannot fill region with short value for data type " + getDataType());
        }
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, int value) {
        switch (getDataType()) {
            case INT32 -> dataContainer.fillRegion(startFlatIndex, endFlatIndex, value);
            case INT64 -> dataContainer.fillRegion(startFlatIndex, endFlatIndex, (long) value);
            case FLOAT32 -> dataContainer.fillRegion(startFlatIndex, endFlatIndex, (float) value);
            case FLOAT64 -> dataContainer.fillRegion(startFlatIndex, endFlatIndex, (double) value);
            default ->
                    throw new IllegalArgumentException("Cannot fill region with int value for data type " + getDataType());
        }
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, long value) {
        switch (getDataType()) {
            case INT64 -> dataContainer.fillRegion(startFlatIndex, endFlatIndex, value);
            case FLOAT32 -> dataContainer.fillRegion(startFlatIndex, endFlatIndex, (float) value);
            case FLOAT64 -> dataContainer.fillRegion(startFlatIndex, endFlatIndex, (double) value);
            default ->
                    throw new IllegalArgumentException("Cannot fill region with long value for data type " + getDataType());
        }
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, float value) {
        switch (getDataType()) {
            case FLOAT32 -> dataContainer.fillRegion(startFlatIndex, endFlatIndex, value);
            case FLOAT64 -> dataContainer.fillRegion(startFlatIndex, endFlatIndex, (double) value);
            default ->
                    throw new IllegalArgumentException("Cannot fill region with float value for data type " + getDataType());
        }
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, double value) {
        if (getDataType() != DataType.FLOAT64) {
            throw new IllegalArgumentException("Cannot fill region with double value for data type " + getDataType());
        }
        dataContainer.fillRegion(startFlatIndex, endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, boolean value) {
        this.dataContainer.fillRegion(startFlatIndex, endFlatIndex, value);
    }

    @Override
    public @NotNull DirectMemoryHandle getContentsAsDirectMemory() {
        return this.dataContainer.getAsDirectBuffer();
    }


    @Override
    public @NotNull DirectMemoryHandle getContentsAsDirectMemory(long startFlatIndex, long endFlatIndex) {
        long nElements = getNumberOfElements();
        if (startFlatIndex < 0 || endFlatIndex > nElements) {
            throw new IndexOutOfBoundsException("Index out of bounds: " + startFlatIndex + " to " + endFlatIndex + " (data container length " + nElements + ")");
        }
        if (startFlatIndex >= endFlatIndex) {
            throw new IllegalArgumentException("startFlatIndex must be less than endFlatIndex");
        }
        return this.dataContainer.getAsDirectBuffer(Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex));
    }


    @Override
    public void readFrom(@NotNull InputStream inputStream) throws IOException {
        DirectMemoryHandle contents = getContentsAsDirectMemory();
        ByteBuffer byteBuffer = contents.asByteBuffer().order(ByteOrder.BIG_ENDIAN);
        byte[] copyBuffer = new byte[33554432]; //32MB
        while (byteBuffer.hasRemaining()) {
            int toCopy = Math.min(byteBuffer.remaining(), copyBuffer.length);
            int read = inputStream.read(copyBuffer, 0, toCopy);
            if (read == -1) {
                throw new IOException("Unexpected end of stream");
            }
            byteBuffer.put(copyBuffer, 0, read);
        }
    }

    @Override
    public @NotNull ISciCoreBackend getSciCoreBackend() {
        return this.backend;
    }

    @NotNull
    public GenCPUTensorDataContainer getDataContainer() {
        return dataContainer;
    }

    @Override
    public void dispose() {
        super.dispose();
        this.dataContainer.decRc();
        this.dataContainer.dispose();
    }
}
