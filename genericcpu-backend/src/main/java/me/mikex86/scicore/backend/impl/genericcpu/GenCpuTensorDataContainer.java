package me.mikex86.scicore.backend.impl.genericcpu;

import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.memory.DirectMemoryManager;
import me.mikex86.scicore.tensor.data.ITensorDataContainer;
import org.jetbrains.annotations.NotNull;
import org.lwjgl.system.MemoryUtil;

import java.nio.*;

import static java.lang.Math.min;

public class GenCpuTensorDataContainer implements ITensorDataContainer {

    @NotNull
    private final DirectMemoryHandle memoryHandle;

    private final long dataSize;

    @NotNull
    private final DataType dataType;

    @NotNull
    private final DirectMemoryManager memoryManager;

    public GenCpuTensorDataContainer(@NotNull DirectMemoryManager memoryManager, long nElements, @NotNull DataType dataType) {
        this.memoryManager = memoryManager;
        long nBytes = dataType.getSizeOf(nElements);
        this.memoryHandle = memoryManager.calloc(nBytes);
        this.dataSize = nBytes;
        this.dataType = dataType;
    }

    @Override
    public byte getInt8Flat(long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex;
        return MemoryUtil.memGetByte(finalPtr);
    }

    @Override
    public void getInt8Flat(byte value, long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex;
        MemoryUtil.memPutByte(finalPtr, value);
    }

    @Override
    public short getInt16Flat(long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 2;
        return MemoryUtil.memGetShort(finalPtr);
    }

    @Override
    public void setInt16Flat(short value, long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 2;
        MemoryUtil.memPutShort(finalPtr, value);
    }

    @Override
    public int getInt32Flat(long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 4;
        return MemoryUtil.memGetInt(finalPtr);
    }

    @Override
    public void setInt32Flat(int value, long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 4;
        MemoryUtil.memPutInt(finalPtr, value);
    }

    @Override
    public long getInt64Flat(long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 8;
        return MemoryUtil.memGetLong(finalPtr);
    }

    @Override
    public void setInt64Flat(long value, long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 8;
        MemoryUtil.memPutLong(finalPtr, value);
    }

    @Override
    public float getFloat32Flat(long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 4;
        return MemoryUtil.memGetFloat(finalPtr);
    }

    @Override
    public void setFloat32Flat(float value, long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 4;
        MemoryUtil.memPutFloat(finalPtr, value);
    }

    @Override
    public double getFloat64Flat(long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 8;
        return MemoryUtil.memGetDouble(finalPtr);
    }

    @Override
    public void setFloat64Flat(double value, long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 8;
        MemoryUtil.memPutDouble(finalPtr, value);
    }

    @Override
    public void setBooleanFlat(boolean value, long flatIndex) {
        long byteIndex = flatIndex / 8;
        int bitIndex = (int) (flatIndex % 8);
        byte byteValue = getInt8Flat(byteIndex);
        byteValue = (byte) (byteValue & ~(1 << bitIndex));
        if (value) {
            byteValue = (byte) (byteValue | (1 << bitIndex));
        }
        getInt8Flat(byteValue, byteIndex);
    }

    @Override
    public boolean getBooleanFlat(long flatIndex) {
        long byteIndex = flatIndex / 8;
        int bitIndex = (int) (flatIndex % 8);
        byte byteValue = getInt8Flat(byteIndex);
        return (byteValue & (1 << bitIndex)) != 0;
    }

    @Override
    public void setContents(@NotNull ByteBuffer buffer) {
        if (buffer.remaining() > this.dataSize) {
            throw new IllegalArgumentException("Cannot set contents of data container of size " + this.dataSize + " with buffer of size " + buffer.remaining());
        }
        if (!buffer.isDirect()) {
            DirectMemoryHandle memoryHandle = memoryManager.alloc(buffer.capacity());
            ByteBuffer directBuffer = memoryHandle.asByteBuffer();
            directBuffer.put(buffer);
            directBuffer.flip();
            long bufferPtr = memoryHandle.getNativePtr();
            MemoryUtil.memCopy(bufferPtr, this.memoryHandle.getNativePtr(), directBuffer.capacity());
            memoryHandle.free();
        } else {
            long bufferPtr = MemoryUtil.memAddress(buffer);
            MemoryUtil.memCopy(bufferPtr, this.memoryHandle.getNativePtr(), buffer.capacity());
        }
    }

    @Override
    public void setContents(@NotNull ShortBuffer buffer) {
        if (buffer.remaining() > this.dataSize / Short.BYTES) {
            throw new IllegalArgumentException("Cannot set contents of data container of size " + this.dataSize + " with buffer of size " + buffer.remaining());
        }
        long nBytes = (long) buffer.capacity() * Short.BYTES;
        if (!buffer.isDirect()) {
            DirectMemoryHandle memoryHandle = memoryManager.alloc(nBytes);
            ShortBuffer directBuffer = memoryHandle.asShortBuffer();
            directBuffer.put(buffer);
            directBuffer.flip();
            long bufferPtr = memoryHandle.getNativePtr();
            MemoryUtil.memCopy(bufferPtr, this.memoryHandle.getNativePtr(), nBytes);
            memoryHandle.free();
        } else {
            long bufferPtr = MemoryUtil.memAddress(buffer);
            MemoryUtil.memCopy(bufferPtr, this.memoryHandle.getNativePtr(), nBytes);
        }
    }

    @Override
    public void setContents(@NotNull IntBuffer buffer) {
        if (buffer.remaining() > this.dataSize / Integer.BYTES) {
            throw new IllegalArgumentException("Cannot set contents of data container of size " + this.dataSize + " with buffer of size " + buffer.remaining());
        }
        long nBytes = (long) buffer.capacity() * Integer.BYTES;
        if (!buffer.isDirect()) {
            DirectMemoryHandle memoryHandle = memoryManager.alloc(nBytes);
            IntBuffer directBuffer = memoryHandle.asIntBuffer();
            directBuffer.put(buffer);
            directBuffer.flip();
            long bufferPtr = memoryHandle.getNativePtr();
            MemoryUtil.memCopy(bufferPtr, this.memoryHandle.getNativePtr(), nBytes);
            memoryHandle.free();
        } else {
            long bufferPtr = MemoryUtil.memAddress(buffer);
            MemoryUtil.memCopy(bufferPtr, this.memoryHandle.getNativePtr(), nBytes);
        }
    }

    @Override
    public void setContents(@NotNull LongBuffer buffer) {
        if (buffer.remaining() > this.dataSize / Long.BYTES) {
            throw new IllegalArgumentException("Cannot set contents of data container of size " + this.dataSize + " with buffer of size " + buffer.remaining());
        }
        long nBytes = (long) buffer.capacity() * Long.BYTES;
        if (!buffer.isDirect()) {
            DirectMemoryHandle memoryHandle = memoryManager.alloc(nBytes);
            LongBuffer directBuffer = memoryHandle.asLongBuffer();
            directBuffer.put(buffer);
            directBuffer.flip();
            long bufferPtr = memoryHandle.getNativePtr();
            MemoryUtil.memCopy(bufferPtr, this.memoryHandle.getNativePtr(), nBytes);
            memoryHandle.free();
        } else {
            long bufferPtr = MemoryUtil.memAddress(buffer);
            MemoryUtil.memCopy(bufferPtr, this.memoryHandle.getNativePtr(), nBytes);
        }
    }

    @Override
    public void setContents(@NotNull FloatBuffer buffer) {
        if (buffer.remaining() > this.dataSize / Float.BYTES) {
            throw new IllegalArgumentException("Cannot set contents of data container of size " + this.dataSize + " with buffer of size " + buffer.remaining());
        }
        long nBytes = (long) buffer.capacity() * Float.BYTES;
        if (!buffer.isDirect()) {
            DirectMemoryHandle memoryHandle = memoryManager.alloc(nBytes);
            FloatBuffer directBuffer = memoryHandle.asFloatBuffer();
            directBuffer.put(buffer);
            directBuffer.flip();
            long bufferPtr = memoryHandle.getNativePtr();
            MemoryUtil.memCopy(bufferPtr, this.memoryHandle.getNativePtr(), nBytes);
            memoryHandle.free();
        } else {
            long bufferPtr = MemoryUtil.memAddress(buffer);
            MemoryUtil.memCopy(bufferPtr, this.memoryHandle.getNativePtr(), nBytes);
        }
    }

    @Override
    public void setContents(@NotNull DoubleBuffer buffer) {
        if (buffer.remaining() > this.dataSize / Double.BYTES) {
            throw new IllegalArgumentException("Cannot set contents of data container of size " + this.dataSize + " with buffer of size " + buffer.remaining());
        }
        long nBytes = (long) buffer.capacity() * Double.BYTES;
        if (!buffer.isDirect()) {
            DirectMemoryHandle memoryHandle = memoryManager.alloc(nBytes);
            DoubleBuffer directBuffer = memoryHandle.asDoubleBuffer();
            directBuffer.put(buffer);
            directBuffer.flip();
            long bufferPtr = memoryHandle.getNativePtr();
            MemoryUtil.memCopy(bufferPtr, this.memoryHandle.getNativePtr(), nBytes);
            memoryHandle.free();
        } else {
            long bufferPtr = MemoryUtil.memAddress(buffer);
            MemoryUtil.memCopy(bufferPtr, this.memoryHandle.getNativePtr(), nBytes);
        }
    }

    @Override
    public void setContents(boolean @NotNull [] data) {
        if (data.length > this.dataSize * 8) {
            throw new IllegalArgumentException("Cannot set contents of data container of size " + this.dataSize * 8 + " bits with buffer of size " + data.length + " bits");
        }
        for (int i = 0; i < data.length; i++) {
            int byteIndex = i / 8;
            int bitIndex = i % 8;
            byte byteValue = getInt8Flat(byteIndex);
            if (data[i]) {
                byteValue = (byte) (byteValue | (1 << bitIndex));
            } else {
                byteValue = (byte) (byteValue & ~(1 << bitIndex));
            }
            getInt8Flat(byteValue, byteIndex);
        }
    }

    @Override
    public void fill(byte value) {
        MemoryUtil.memSet(this.memoryHandle.getNativePtr(), value & 0xFF, this.dataSize);
    }

    @Override
    public void fill(short value) {
        if (this.dataSize % Short.BYTES != 0) {
            throw new IllegalArgumentException("Cannot fill data container of size " + this.dataSize + " with short value");
        }
        ShortBuffer buffer = this.memoryHandle.asShortBuffer();
        for (int i = 0; i < buffer.capacity(); i++) {
            buffer.put(i, value);
        }
        buffer.flip();
    }

    @Override
    public void fill(int value) {
        if (this.dataSize % Integer.BYTES != 0) {
            throw new IllegalArgumentException("Cannot fill data container of size " + this.dataSize + " with int value");
        }
        IntBuffer buffer = this.memoryHandle.asIntBuffer();
        for (int i = 0; i < buffer.capacity(); i++) {
            buffer.put(i, value);
        }
        buffer.flip();
    }

    @Override
    public void fill(long value) {
        if (this.dataSize % Long.BYTES != 0) {
            throw new IllegalArgumentException("Cannot fill data container of size " + this.dataSize + " with long value");
        }
        LongBuffer buffer = this.memoryHandle.asLongBuffer();
        for (int i = 0; i < buffer.capacity(); i++) {
            buffer.put(i, value);
        }
        buffer.flip();
    }

    @Override
    public void fill(float value) {
        if (this.dataSize % Float.BYTES != 0) {
            throw new IllegalArgumentException("Cannot fill data container of size " + this.dataSize + " with float value");
        }
        FloatBuffer buffer = this.memoryHandle.asFloatBuffer();
        for (int i = 0; i < buffer.capacity(); i++) {
            buffer.put(i, value);
        }
        buffer.flip();
    }

    @Override
    public void fill(double value) {
        if (this.dataSize % Double.BYTES != 0) {
            throw new IllegalArgumentException("Cannot fill data container of size " + this.dataSize + " with double value");
        }
        DoubleBuffer buffer = this.memoryHandle.asDoubleBuffer();
        for (int i = 0; i < buffer.capacity(); i++) {
            buffer.put(i, value);
        }
        buffer.flip();
    }

    @Override
    public void fill(boolean value) {
        byte byteValue = value ? (byte) 0xFF : (byte) 0x00;
        fill(byteValue);
    }

    @Override
    @NotNull
    public DirectMemoryHandle getAsDirectBuffer(long startFlatIndex, long endFlatIndex) {
        long containerSize = this.dataSize;
        if (startFlatIndex < 0 || endFlatIndex > containerSize) {
            throw new IndexOutOfBoundsException("Index out of bounds: " + startFlatIndex + " to " + endFlatIndex + " (data container length " + containerSize + ")");
        }
        if (startFlatIndex >= endFlatIndex) {
            throw new IllegalArgumentException("startFlatIndex must be less than endFlatIndex");
        }
        long nElements = endFlatIndex - startFlatIndex;
        return this.memoryHandle.offset(startFlatIndex, nElements, dataType);
    }

    @Override
    @NotNull
    public DirectMemoryHandle getAsDirectBuffer() {
        return this.memoryHandle.createReference();
    }

    @Override
    @NotNull
    public DataType getDataType() {
        return dataType;
    }

    @NotNull
    public DirectMemoryHandle getMemoryHandle() {
        return memoryHandle;
    }

    @Override
    public long getDataSize() {
        return dataSize;
    }

    @Override
    public void dispose() {
        this.memoryHandle.free();
    }
}
