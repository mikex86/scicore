package me.mikex86.scicore.backend.impl.genericcpu;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.memory.DirectMemoryManager;
import org.jetbrains.annotations.NotNull;
import org.lwjgl.system.MemoryUtil;

import java.nio.*;

import static java.lang.Math.min;

public class GenCpuTensorDataContainer {

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

    public byte getByteFlat(long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex;
        return MemoryUtil.memGetByte(finalPtr);
    }

    public void setByteFlat(byte value, long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex;
        MemoryUtil.memPutByte(finalPtr, value);
    }

    public short getInt16Flat(long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 2;
        return MemoryUtil.memGetShort(finalPtr);
    }

    public void setInt16Flat(short value, long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 2;
        MemoryUtil.memPutShort(finalPtr, value);
    }

    public int getInt32Flat(long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 4;
        return MemoryUtil.memGetInt(finalPtr);
    }

    public void setInt32Flat(int value, long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 4;
        MemoryUtil.memPutInt(finalPtr, value);
    }

    public long getInt64Flat(long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 8;
        return MemoryUtil.memGetLong(finalPtr);
    }

    public void setInt64Flat(long value, long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 8;
        MemoryUtil.memPutLong(finalPtr, value);
    }

    public float getFloat32Flat(long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 4;
        return MemoryUtil.memGetFloat(finalPtr);
    }

    public void setFloat32Flat(float value, long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 4;
        MemoryUtil.memPutFloat(finalPtr, value);
    }

    public double getFloat64Flat(long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 8;
        return MemoryUtil.memGetDouble(finalPtr);
    }

    public void setFloat64Flat(double value, long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 8;
        MemoryUtil.memPutDouble(finalPtr, value);
    }

    public void setBooleanFlat(boolean value, long flatIndex) {
        long byteIndex = flatIndex / 8;
        int bitIndex = (int) (flatIndex % 8);
        byte byteValue = getByteFlat(byteIndex);
        byteValue = (byte) (byteValue & ~(1 << bitIndex));
        if (value) {
            byteValue = (byte) (byteValue | (1 << bitIndex));
        }
        setByteFlat(byteValue, byteIndex);
    }

    public boolean getBooleanFlat(long flatIndex) {
        long byteIndex = flatIndex / 8;
        int bitIndex = (int) (flatIndex % 8);
        byte byteValue = getByteFlat(byteIndex);
        return (byteValue & (1 << bitIndex)) != 0;
    }

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

    public void setContents(boolean @NotNull [] data) {
        if (data.length > this.dataSize * 8) {
            throw new IllegalArgumentException("Cannot set contents of data container of size " + this.dataSize * 8 + " bits with buffer of size " + data.length + " bits");
        }
        for (int i = 0; i < data.length; i++) {
            int byteIndex = i / 8;
            int bitIndex = i % 8;
            byte byteValue = getByteFlat(byteIndex);
            if (data[i]) {
                byteValue = (byte) (byteValue | (1 << bitIndex));
            } else {
                byteValue = (byte) (byteValue & ~(1 << bitIndex));
            }
            setByteFlat(byteValue, byteIndex);
        }
    }

    /**
     * Returns a reference memory handle to the internal data buffer in the specified index range.
     *
     * @param startFlatIndex the start index of the data to copy (flat index)
     * @param endFlatIndex   the end index of the data to copy (exclusive, flat index)
     * @return the host byte buffer. Must not be freed.
     */
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
        return this.memoryHandle.offset(startFlatIndex, nElements);
    }

    /**
     * @return a reference memory handle to the internal data buffer. Must not be freed.
     */
    @NotNull
    public DirectMemoryHandle getAsDirectBuffer() {
        return this.memoryHandle.createReference();
    }

    @NotNull
    public DataType getDataType() {
        return dataType;
    }

    @NotNull
    public DirectMemoryHandle getMemoryHandle() {
        return memoryHandle;
    }

    public long getDataSize() {
        return dataSize;
    }
}
