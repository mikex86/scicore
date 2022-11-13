package me.mikex86.scicore.backend.impl.cuda;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.tensor.AbstractTensor;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

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
        return dataContainer.getInt8Flat(flatIndex);
    }

    @Override
    public void setByteFlat(byte value, long flatIndex) {
        dataContainer.getInt8Flat(value, flatIndex);
    }

    @Override
    public short getShortFlat(long flatIndex) {
        return dataContainer.getInt16Flat(flatIndex);
    }

    @Override
    public void setShortFlat(short value, long flatIndex) {
        dataContainer.setInt16Flat(value, flatIndex);
    }

    @Override
    public int getIntFlat(long flatIndex) {
        return dataContainer.getInt32Flat(flatIndex);
    }

    @Override
    public void setIntFlat(int value, long flatIndex) {
        dataContainer.setInt32Flat(value, flatIndex);
    }

    @Override
    public long getLongFlat(long flatIndex) {
        return dataContainer.getInt64Flat(flatIndex);
    }

    @Override
    public void setLongFlat(long value, long flatIndex) {
        dataContainer.setInt64Flat(value, flatIndex);
    }


    @Override
    public float getFloatFlat(long flatIndex) {
        return dataContainer.getFloat32Flat(flatIndex);
    }

    @Override
    public void setFloatFlat(float value, long flatIndex) {
        dataContainer.setFloat32Flat(value, flatIndex);
    }

    @Override
    public double getDoubleFlat(long flatIndex) {
        return dataContainer.getFloat32Flat(flatIndex);
    }

    @Override
    public void setDoubleFlat(double value, long flatIndex) {
        dataContainer.setFloat64Flat(value, flatIndex);
    }

    @Override
    public boolean getBooleanFlat(long flatIndex) {
        return dataContainer.getBooleanFlat(flatIndex);
    }

    @Override
    public void setBooleanFlat(boolean value, long flatIndex) {
        dataContainer.setBooleanFlat(value, flatIndex);
    }

    @Override
    public @NotNull ITensor copy() {
        throw new UnsupportedOperationException("TODO: implement");
    }

    @Override
    public void setContents(@NotNull ITensor tensor) {
        Validator.assertTrue(ShapeUtils.equals(shape, tensor.getShape()), "Cannot set contents of tensor with shape " + ShapeUtils.toString(shape) + " to tensor with shape " + ShapeUtils.toString(tensor.getShape()));
        Validator.assertTrue(dataType == tensor.getDataType(), "Cannot set contents of tensor with data type " + dataType + " to tensor with data type " + tensor.getDataType());
        // TODO: Handle Lazy Tensors and CUDA Tensor Views more efficiently
        if (tensor instanceof CudaTensor cudaTensor) {
            backend.getCudaMemoryManager().copy(dataContainer.getDeviceMemoryHandle(), cudaTensor.dataContainer.getDeviceMemoryHandle());
        } else {
            DirectMemoryHandle contents = tensor.getContentsAsDirectMemory();
            dataContainer.setContents(contents.asByteBuffer());
            if (contents.canFree()) {
                contents.free();
            }
        }
    }

    @Override
    public void setContents(@NotNull ByteBuffer buffer) {
        Validator.assertTrue(dataType == DataType.INT8, "Cannot set contents of tensor with data type " + dataType);
        this.dataContainer.setContents(buffer);
    }

    @Override
    public void setContents(@NotNull ShortBuffer buffer) {
        Validator.assertTrue(dataType == DataType.INT16, "Cannot set contents of tensor with data type " + dataType);
        this.dataContainer.setContents(buffer);
    }

    @Override
    public void setContents(@NotNull IntBuffer buffer) {
        Validator.assertTrue(dataType == DataType.INT32, "Cannot set contents of tensor with data type " + dataType);
        this.dataContainer.setContents(buffer);
    }

    @Override
    public void setContents(@NotNull LongBuffer buffer) {
        Validator.assertTrue(dataType == DataType.INT64, "Cannot set contents of tensor with data type " + dataType);
        this.dataContainer.setContents(buffer);
    }

    @Override
    public void setContents(@NotNull FloatBuffer buffer) {
        Validator.assertTrue(dataType == DataType.FLOAT32, "Cannot set contents of tensor with data type " + dataType);
        this.dataContainer.setContents(buffer);
    }

    @Override
    public void setContents(@NotNull DoubleBuffer buffer) {
        Validator.assertTrue(dataType == DataType.FLOAT64, "Cannot set contents of tensor with data type " + dataType);
        this.dataContainer.setContents(buffer);
    }

    @Override
    public void setContents(boolean @NotNull [] buffer) {
        Validator.assertTrue(dataType == DataType.BOOLEAN, "Cannot set contents of tensor with data type " + dataType);
        this.dataContainer.setContents(buffer);
    }

    @Override
    public void setContents(long @NotNull [] index, @NotNull ITensor tensor) {
        long flatIndex = ShapeUtils.getFlatIndex(index, this.strides);
        if (tensor instanceof CudaTensor cudaTensor) {
            // device to device copy
            this.dataContainer.setContents(cudaTensor.dataContainer.getDeviceMemoryHandle(), flatIndex);
        } else {
            // general copy
            DirectMemoryHandle memoryHandle = tensor.getContentsAsDirectMemory();
            ByteBuffer hostBuffer = memoryHandle.asByteBuffer();
            this.dataContainer.setContents(hostBuffer, flatIndex);
            if (memoryHandle.canFree()) {
                memoryHandle.free();
            }
        }
    }

    @Override
    public void fill(byte value) {
        this.dataContainer.fill(value);
    }

    @Override
    public void fill(short value) {
        this.dataContainer.fill(value);
    }

    @Override
    public void fill(int value) {
        this.dataContainer.fill(value);
    }

    @Override
    public void fill(long value) {
        this.dataContainer.fill(value);
    }

    @Override
    public void fill(float value) {
        this.dataContainer.fill(value);
    }

    @Override
    public void fill(double value) {
        this.dataContainer.fill(value);
    }

    @Override
    public void fill(boolean value) {
        this.dataContainer.fill(value);
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
