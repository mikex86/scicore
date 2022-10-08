package me.mikex86.scicore.backend.impl.genericcpu;

import me.mikex86.scicore.*;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.utils.Pair;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.nio.*;

public class GenCPUTensor extends AbstractTensor implements ITensor {

    @NotNull
    private final GenCpuTensorDataContainer dataContainer;

    private final long @NotNull [] strides;

    @NotNull
    private final GenCPUBackend backend;

    private final long @NotNull [] shape;

    public GenCPUTensor(@NotNull GenCPUBackend backend, @NotNull DataType dataType, long @NotNull [] shape) {
        this.numElements = ShapeUtils.getNumElements(shape);
        this.dataContainer = new GenCpuTensorDataContainer(backend.getMemoryManager(), this.numElements, dataType);
        this.strides = ShapeUtils.makeStrides(shape);
        this.shape = shape;
        this.backend = backend;
    }

    GenCPUTensor(@NotNull GenCPUBackend backend, @NotNull GenCpuTensorDataContainer dataContainer, long @NotNull [] shape) {
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
    public short getShort(long @NotNull [] indices) {
        long index = ShapeUtils.getFlatIndex(indices, this.strides);
        return this.dataContainer.getInt16Flat(index);
    }

    @Override
    public int getInt(long @NotNull [] indices) {
        long index = ShapeUtils.getFlatIndex(indices, this.strides);
        return this.dataContainer.getInt32Flat(index);
    }

    @Override
    public long getLong(long @NotNull [] indices) {
        long index = ShapeUtils.getFlatIndex(indices, this.strides);
        return this.dataContainer.getInt64Flat(index);
    }

    @Override
    public boolean getBooleanFlat(long flatIndex) {
        return this.dataContainer.getBooleanFlat(flatIndex);
    }

    @Override
    public void setBooleanFlat(boolean value, long flatIndex) {
        this.dataContainer.setBooleanFlat(value, flatIndex);
    }

    @Override
    public byte getByteFlat(long flatIndex) {
        return this.dataContainer.getByteFlat(flatIndex);
    }

    @Override
    public void setByteFlat(byte value, long flatIndex) {
        this.dataContainer.setByteFlat(value, flatIndex);
    }

    @Override
    public short getShortFlat(long flatIndex) {
        return this.dataContainer.getInt16Flat(flatIndex);
    }

    @Override
    public void setShortFlat(short value, long flatIndex) {
        this.dataContainer.setInt16Flat(value, flatIndex);
    }

    @Override
    public int getIntFlat(long flatIndex) {
        return this.dataContainer.getInt32Flat(flatIndex);
    }

    @Override
    public void setIntFlat(int value, long flatIndex) {
        this.dataContainer.setInt32Flat(value, flatIndex);
    }

    @Override
    public long getLongFlat(long flatIndex) {
        return this.dataContainer.getInt64Flat(flatIndex);
    }

    @Override
    public void setLongFlat(long value, long flatIndex) {
        this.dataContainer.setInt64Flat(value, flatIndex);
    }

    @Override
    public float getFloatFlat(long flatIndex) {
        return this.dataContainer.getFloat32Flat(flatIndex);
    }

    @Override
    public void setFloatFlat(float value, long flatIndex) {
        this.dataContainer.setFloat32Flat(value, flatIndex);
    }

    @Override
    public double getDoubleFlat(long flatIndex) {
        return this.dataContainer.getFloat64Flat(flatIndex);
    }

    @Override
    public void setDoubleFlat(double value, long flatIndex) {
        this.dataContainer.setFloat64Flat(value, flatIndex);
    }

    @Override
    public @NotNull ITensor copy() {
        ITensor copy = new GenCPUTensor(backend, getDataType(), getShape());
        copy.setContents(this);
        return copy;
    }

    @Override
    public void setContents(@NotNull ITensor tensor) {
        if (tensor.getDataType() != getDataType()) {
            throw new IllegalArgumentException("Cannot copy tensor with different data type");
        }
        if (!ShapeUtils.equals(tensor.getShape(), getShape())) {
            throw new IllegalArgumentException("Cannot copy tensor with different shape");
        }
        DirectMemoryHandle directMemory = tensor.getContentsAsDirectMemory();
        this.dataContainer.setContents(directMemory.asByteBuffer());
        if (directMemory.canFree()) {
            directMemory.free();
        }
    }

    @Override
    public void setContents(@NotNull ByteBuffer buffer) {
        this.dataContainer.setContents(buffer);
    }

    @Override
    public void setContents(@NotNull ShortBuffer buffer) {
        this.dataContainer.setContents(buffer);
    }

    @Override
    public void setContents(@NotNull IntBuffer buffer) {
        this.dataContainer.setContents(buffer);
    }

    @Override
    public void setContents(@NotNull LongBuffer buffer) {
        this.dataContainer.setContents(buffer);
    }

    @Override
    public void setContents(@NotNull FloatBuffer buffer) {
        this.dataContainer.setContents(buffer);
    }

    @Override
    public void setContents(@NotNull DoubleBuffer buffer) {
        this.dataContainer.setContents(buffer);
    }

    @Override
    public void setContents(boolean @NotNull [] buffer) {
        this.dataContainer.setContents(buffer);
    }

    @Override
    public void setContents(long @NotNull [] index, @NotNull ITensor tensor, boolean useView) {
        // TODO: implement useView
        // General copy
        long startIndex = ShapeUtils.getFlatIndex(index, this.strides);
        long nElementsToCopy = tensor.getNumberOfElements();
        for (long i = 0; i < nElementsToCopy; i++) {
            switch (this.getDataType()) {
                case INT8 -> setByteFlat(tensor.getByteFlat(i), startIndex + i);
                case INT16 -> setShortFlat(tensor.getShortFlat(i), startIndex + i);
                case INT32 -> setIntFlat(tensor.getIntFlat(i), startIndex + i);
                case INT64 -> setLongFlat(tensor.getLongFlat(i), startIndex + i);
                case FLOAT32 -> setFloatFlat(tensor.getFloatFlat(i), startIndex + i);
                case FLOAT64 -> setDoubleFlat(tensor.getDoubleFlat(i), startIndex + i);
                default -> throw new IllegalArgumentException("Unsupported data type");
            }
        }
    }

    @Override
    public void fill(byte i) {
        long nElements = ShapeUtils.getNumElements(getShape());
        switch (this.getDataType()) {
            case INT8 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setByteFlat(i, j);
                }
            }
            case INT16 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt16Flat(i, j);
                }
            }
            case INT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt32Flat(i, j);
                }
            }
            case INT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt64Flat(i, j);
                }
            }
            case FLOAT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setFloat32Flat(i, j);
                }
            }
            case FLOAT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setFloat64Flat(i, j);
                }
            }
        }
    }

    @Override
    public void fill(short i) {
        long nElements = ShapeUtils.getNumElements(getShape());
        switch (this.getDataType()) {
            case INT8 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setByteFlat((byte) i, j);
                }
            }
            case INT16 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt16Flat(i, j);
                }
            }
            case INT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt32Flat(i, j);
                }
            }
            case INT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt64Flat(i, j);
                }
            }
            case FLOAT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setFloat32Flat(i, j);
                }
            }
            case FLOAT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setFloat64Flat(i, j);
                }
            }
        }
    }

    @Override
    public void fill(int i) {
        long nElements = ShapeUtils.getNumElements(getShape());
        switch (this.getDataType()) {
            case INT8 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setByteFlat((byte) i, j);
                }
            }
            case INT16 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt16Flat((short) i, j);
                }
            }
            case INT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt32Flat(i, j);
                }
            }
            case INT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt64Flat(i, j);
                }
            }
            case FLOAT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setFloat32Flat(i, j);
                }
            }
            case FLOAT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setFloat64Flat(i, j);
                }
            }
        }
    }

    @Override
    public void fill(long i) {
        long nElements = ShapeUtils.getNumElements(getShape());
        switch (this.getDataType()) {
            case INT8 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setByteFlat((byte) i, j);
                }
            }
            case INT16 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt16Flat((short) i, j);
                }
            }
            case INT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt32Flat((int) i, j);
                }
            }
            case INT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt64Flat(i, j);
                }
            }
            case FLOAT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setFloat32Flat(i, j);
                }
            }
            case FLOAT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setFloat64Flat(i, j);
                }
            }
        }
    }

    @Override
    public void fill(float f) {
        long nElements = ShapeUtils.getNumElements(getShape());
        switch (this.getDataType()) {
            case INT8 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setByteFlat((byte) f, j);
                }
            }
            case INT16 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt16Flat((short) f, j);
                }
            }
            case INT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt32Flat((int) f, j);
                }
            }
            case INT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt64Flat((long) f, j);
                }
            }
            case FLOAT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setFloat32Flat(f, j);
                }
            }
            case FLOAT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setFloat64Flat(f, j);
                }
            }
        }
    }

    @Override
    public void fill(double d) {
        long nElements = ShapeUtils.getNumElements(getShape());
        switch (this.getDataType()) {
            case INT8 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setByteFlat((byte) d, j);
                }
            }
            case INT16 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt16Flat((short) d, j);
                }
            }
            case INT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt32Flat((int) d, j);
                }
            }
            case INT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt64Flat((long) d, j);
                }
            }
            case FLOAT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setFloat32Flat((float) d, j);
                }
            }
            case FLOAT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setFloat64Flat(d, j);
                }
            }
        }
    }

    @Override
    public void fill(boolean value) {
        long nElements = ShapeUtils.getNumElements(getShape());
        switch (this.getDataType()) {
            case INT8 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setByteFlat((byte) (value ? 1 : 0), j);
                }
            }
            case INT16 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt16Flat((short) (value ? 1 : 0), j);
                }
            }
            case INT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt32Flat(value ? 1 : 0, j);
                }
            }
            case INT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt64Flat(value ? 1 : 0, j);
                }
            }
            case FLOAT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setFloat32Flat(value ? 1 : 0, j);
                }
            }
            case FLOAT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setFloat64Flat(value ? 1 : 0, j);
                }
            }
        }
    }


    @Override
    public @NotNull DirectMemoryHandle getContentsAsDirectMemory() {
        long nElements = getNumberOfElements();
        if (nElements > Integer.MAX_VALUE) {
            throw new UnsupportedOperationException("JvmTensors cannot have more than Integer.MAX_VALUE elements");
        }
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
        if (startFlatIndex > Integer.MAX_VALUE || endFlatIndex > Integer.MAX_VALUE) {
            throw new UnsupportedOperationException("JvmTensors cannot have more than Integer.MAX_VALUE elements");
        }
        return this.dataContainer.getAsDirectBuffer(Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex));
    }

    @Override
    public @NotNull ISciCoreBackend getSciCoreBackend() {
        return this.backend;
    }

    @NotNull
    public GenCpuTensorDataContainer getDataContainer() {
        return dataContainer;
    }
}
