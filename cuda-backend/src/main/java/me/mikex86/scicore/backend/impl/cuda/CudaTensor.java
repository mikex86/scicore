package me.mikex86.scicore.backend.impl.cuda;

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

public class CudaTensor extends AbstractTensor {

    @NotNull
    private final CudaBackend backend;

    @NotNull
    private final DataType dataType;

    private final long @NotNull [] shape;

    private final long @NotNull [] strides;

    @NotNull
    private final CudaDataContainer dataContainer;

    public CudaTensor(@NotNull CudaBackend backend, @NotNull DataType dataType, long @NotNull [] shape) {
        this.numElements = ShapeUtils.getNumElements(shape);
        this.backend = backend;
        this.dataType = dataType;
        this.shape = shape;
        this.strides = ShapeUtils.makeStrides(shape);
        this.dataContainer = new CudaDataContainer(backend, backend.getCudaMemoryManager(), this.numElements, dataType);
    }

    @Override
    public @NotNull DataType getDataType() {
        return dataType;
    }

    @Override
    public long @NotNull [] getShape() {
        return shape;
    }

    @Override
    public long @NotNull [] getStrides() {
        return strides;
    }

    @Override
    public byte getByteFlat(long flatIndex) {
        validateDataType(DataType.INT8);
        return dataContainer.getInt8Flat(flatIndex);
    }

    @Override
    public void setByteFlat(byte value, long flatIndex) {
        validateDataType(DataType.INT8);
        dataContainer.setInt8Flat(flatIndex, value);
    }

    @Override
    public short getShortFlat(long flatIndex) {
        validateDataType(DataType.INT16);
        return dataContainer.getInt16Flat(flatIndex);
    }

    @Override
    public void setShortFlat(short value, long flatIndex) {
        validateDataType(DataType.INT16);
        dataContainer.setInt16Flat(flatIndex, value);
    }

    @Override
    public int getIntFlat(long flatIndex) {
        validateDataType(DataType.INT32);
        return dataContainer.getInt32Flat(flatIndex);
    }

    @Override
    public void setIntFlat(int value, long flatIndex) {
        validateDataType(DataType.INT32);
        dataContainer.setInt32Flat(flatIndex, value);
    }

    @Override
    public long getLongFlat(long flatIndex) {
        validateDataType(DataType.INT64);
        return dataContainer.getInt64Flat(flatIndex);
    }

    @Override
    public void setLongFlat(long value, long flatIndex) {
        validateDataType(DataType.INT64);
        dataContainer.setInt64Flat(flatIndex, value);
    }


    @Override
    public float getFloatFlat(long flatIndex) {
        validateDataType(DataType.FLOAT32);
        return dataContainer.getFloat32Flat(flatIndex);
    }

    @Override
    public void setFloatFlat(float value, long flatIndex) {
        validateDataType(DataType.FLOAT32);
        dataContainer.setFloat32Flat(flatIndex, value);
    }

    @Override
    public double getDoubleFlat(long flatIndex) {
        validateDataType(DataType.FLOAT64);
        return dataContainer.getFloat32Flat(flatIndex);
    }

    @Override
    public void setDoubleFlat(double value, long flatIndex) {
        validateDataType(DataType.FLOAT64);
        dataContainer.setFloat64Flat(flatIndex, value);
    }

    @Override
    public boolean getBooleanFlat(long flatIndex) {
        validateDataType(DataType.BOOLEAN);
        return dataContainer.getBooleanFlat(flatIndex);
    }

    @Override
    public void setBooleanFlat(boolean value, long flatIndex) {
        validateDataType(DataType.BOOLEAN);
        dataContainer.setBooleanFlat(flatIndex, value);
    }

    @Override
    public @NotNull ITensor copy() {
        throw new UnsupportedOperationException("TODO: implement");
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull ITensor tensor) {
        validateDataType(tensor.getDataType());
        // TODO: Handle Lazy Tensors and CUDA Tensor Views more efficiently
        if (tensor instanceof CudaTensor cudaTensor) {
            // device to device copy
            this.dataContainer.setContents(startFlatIndex, cudaTensor.dataContainer.getDeviceMemoryHandle());
        } else {
            // general copy
            DirectMemoryHandle memoryHandle = tensor.getContentsAsDirectMemory();
            ByteBuffer hostBuffer = memoryHandle.asByteBuffer();
            this.dataContainer.setContents(startFlatIndex, hostBuffer);
            if (memoryHandle.canFree()) {
                memoryHandle.free();
            }
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
        validateDataType(DataType.INT32);
        this.dataContainer.setContents(startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull LongBuffer buffer) {
        validateDataType(DataType.INT64);
        this.dataContainer.setContents(startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull FloatBuffer buffer) {
        validateDataType(DataType.FLOAT32);
        this.dataContainer.setContents(startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull DoubleBuffer buffer) {
        validateDataType(DataType.FLOAT64);
        this.dataContainer.setContents(startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, boolean @NotNull [] buffer) {
        validateDataType(DataType.BOOLEAN);
        this.dataContainer.setContents(startFlatIndex, buffer);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, byte value) {
        this.dataContainer.fillRegion(startFlatIndex, endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, short value) {
        validateDataType(DataType.INT16);
        this.dataContainer.fillRegion(startFlatIndex, endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, int value) {
        validateDataType(DataType.INT32);
        this.dataContainer.fillRegion(startFlatIndex, endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, long value) {
        validateDataType(DataType.INT64);
        this.dataContainer.fillRegion(startFlatIndex, endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, float value) {
        validateDataType(DataType.FLOAT32);
        this.dataContainer.fillRegion(startFlatIndex, endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, double value) {
        validateDataType(DataType.FLOAT64);
        this.dataContainer.fillRegion(startFlatIndex, endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, boolean value) {
        validateDataType(DataType.BOOLEAN);
        this.dataContainer.fillRegion(startFlatIndex, endFlatIndex, value);
    }

    @Override
    public void readFrom(@NotNull InputStream inputStream) throws IOException {
        // TODO: WE COULD COPY THIS IN CHUNKS TO THE GPU TO SAVE RAM
        byte[] bytes = inputStream.readAllBytes();
        DirectMemoryHandle memoryHandle = backend.getDirectMemoryManager().alloc(bytes.length);
        ByteBuffer byteBuffer = memoryHandle.asByteBuffer();
        byteBuffer.put(bytes);
        this.dataContainer.setContents(byteBuffer);
        memoryHandle.free();
    }

    @Override
    public @NotNull ISciCoreBackend getSciCoreBackend() {
        return backend;
    }

    @Override
    public @NotNull DirectMemoryHandle getContentsAsDirectMemory() {
        return dataContainer.getAsDirectBuffer(0, getNumberOfElements());
    }

    @Override
    public @NotNull DirectMemoryHandle getContentsAsDirectMemory(long startFlatIndex, long endFlatIndex) {
        return dataContainer.getAsDirectBuffer(startFlatIndex, endFlatIndex);
    }

    @NotNull
    public CudaDataContainer getDataContainer() {
        return dataContainer;
    }

    @Override
    public void dispose() {
        super.dispose();
        this.dataContainer.dispose();
    }
}
