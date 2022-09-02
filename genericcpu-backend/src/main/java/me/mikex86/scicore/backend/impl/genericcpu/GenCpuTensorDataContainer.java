package me.mikex86.scicore.backend.impl.genericcpu;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.backend.impl.genericcpu.mem.GenCpuMemoryManager;
import org.jetbrains.annotations.NotNull;
import org.lwjgl.system.MemoryUtil;

import java.nio.*;

import static java.lang.Math.min;

class GenCpuTensorDataContainer {

    private final long dataPtr;

    private final long dataSize;

    @NotNull
    private final DataType dataType;

    @NotNull
    private final GenCpuMemoryManager memoryManager;

    public GenCpuTensorDataContainer(@NotNull GenCpuMemoryManager memoryManager, long nElements, @NotNull DataType dataType) {
        this.memoryManager = memoryManager;
        long nBytes = dataType.getSizeOf(nElements);
        this.dataPtr = memoryManager.alloc(nBytes);
        this.dataSize = nBytes;
        this.dataType = dataType;
    }

    public byte getByteFlat(long flatIndex) {
        long finalPtr = dataPtr + flatIndex;
        return MemoryUtil.memGetByte(finalPtr);
    }

    public void setByteFlat(byte value, long flatIndex) {
        long finalPtr = dataPtr + flatIndex;
        MemoryUtil.memPutByte(finalPtr, value);
    }

    public short getInt16Flat(long flatIndex) {
        long finalPtr = dataPtr + flatIndex * 2;
        return MemoryUtil.memGetShort(finalPtr);
    }

    public void setInt16Flat(short value, long flatIndex) {
        long finalPtr = dataPtr + flatIndex * 2;
        MemoryUtil.memPutShort(finalPtr, value);
    }

    public int getInt32Flat(long flatIndex) {
        long finalPtr = dataPtr + flatIndex * 4;
        return MemoryUtil.memGetInt(finalPtr);
    }

    public void setInt32Flat(int value, long flatIndex) {
        long finalPtr = dataPtr + flatIndex * 4;
        MemoryUtil.memPutInt(finalPtr, value);
    }

    public long getInt64Flat(long flatIndex) {
        long finalPtr = dataPtr + flatIndex * 8;
        return MemoryUtil.memGetLong(finalPtr);
    }

    public void setInt64Flat(long value, long flatIndex) {
        long finalPtr = dataPtr + flatIndex * 8;
        MemoryUtil.memPutLong(finalPtr, value);
    }

    public float getFloat32Flat(long flatIndex) {
        long finalPtr = dataPtr + flatIndex * 4;
        return MemoryUtil.memGetFloat(finalPtr);
    }

    public void setFloat32Flat(float value, long flatIndex) {
        long finalPtr = dataPtr + flatIndex * 4;
        MemoryUtil.memPutFloat(finalPtr, value);
    }

    public double getFloat64Flat(long flatIndex) {
        long finalPtr = dataPtr + flatIndex * 8;
        return MemoryUtil.memGetDouble(finalPtr);
    }

    public void setFloat64Flat(double value, long flatIndex) {
        long finalPtr = dataPtr + flatIndex * 8;
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
            ByteBuffer directBuffer = memoryManager.allocBuffer(buffer.capacity());
            directBuffer.put(buffer);
            directBuffer.flip();
            long bufferPtr = MemoryUtil.memAddress(directBuffer);
            MemoryUtil.memCopy(bufferPtr, this.dataPtr, directBuffer.capacity());
            memoryManager.free(directBuffer);
        } else {
            long bufferPtr = MemoryUtil.memAddress(buffer);
            MemoryUtil.memCopy(bufferPtr, this.dataPtr, buffer.capacity());
        }
    }

    public void setContents(@NotNull ShortBuffer buffer) {
        if (buffer.remaining() > this.dataSize / Short.BYTES) {
            throw new IllegalArgumentException("Cannot set contents of data container of size " + this.dataSize + " with buffer of size " + buffer.remaining());
        }
        if (!buffer.isDirect()) {
            ShortBuffer directBuffer = memoryManager.allocShortBuffer(buffer.capacity());
            directBuffer.put(buffer);
            directBuffer.flip();
            long bufferPtr = MemoryUtil.memAddress(directBuffer);
            MemoryUtil.memCopy(bufferPtr, this.dataPtr, directBuffer.capacity());
            memoryManager.free(directBuffer);
        } else {
            long bufferPtr = MemoryUtil.memAddress(buffer);
            MemoryUtil.memCopy(bufferPtr, this.dataPtr, buffer.capacity());
        }
    }

    public void setContents(@NotNull IntBuffer buffer) {
        if (buffer.remaining() > this.dataSize / Integer.BYTES) {
            throw new IllegalArgumentException("Cannot set contents of data container of size " + this.dataSize + " with buffer of size " + buffer.remaining());
        }
        if (!buffer.isDirect()) {
            IntBuffer directBuffer = memoryManager.allocIntBuffer(buffer.capacity());
            directBuffer.put(buffer);
            directBuffer.flip();
            long bufferPtr = MemoryUtil.memAddress(directBuffer);
            MemoryUtil.memCopy(bufferPtr, this.dataPtr, directBuffer.capacity());
            memoryManager.free(directBuffer);
        } else {
            long bufferPtr = MemoryUtil.memAddress(buffer);
            MemoryUtil.memCopy(bufferPtr, this.dataPtr, buffer.capacity());
        }
    }

    public void setContents(@NotNull LongBuffer buffer) {
        if (buffer.remaining() > this.dataSize / Long.BYTES) {
            throw new IllegalArgumentException("Cannot set contents of data container of size " + this.dataSize + " with buffer of size " + buffer.remaining());
        }
        if (!buffer.isDirect()) {
            LongBuffer directBuffer = memoryManager.allocLongBuffer(buffer.capacity());
            directBuffer.put(buffer);
            directBuffer.flip();
            long bufferPtr = MemoryUtil.memAddress(directBuffer);
            MemoryUtil.memCopy(bufferPtr, this.dataPtr, directBuffer.capacity());
            memoryManager.free(directBuffer);
        } else {
            long bufferPtr = MemoryUtil.memAddress(buffer);
            MemoryUtil.memCopy(bufferPtr, this.dataPtr, buffer.capacity());
        }
    }

    public void setContents(@NotNull FloatBuffer buffer) {
        if (buffer.remaining() > this.dataSize / Float.BYTES) {
            throw new IllegalArgumentException("Cannot set contents of data container of size " + this.dataSize + " with buffer of size " + buffer.remaining());
        }
        if (!buffer.isDirect()) {
            FloatBuffer directBuffer = memoryManager.allocFloatBuffer(buffer.capacity());
            directBuffer.put(buffer);
            directBuffer.flip();
            long bufferPtr = MemoryUtil.memAddress(directBuffer);
            MemoryUtil.memCopy(bufferPtr, this.dataPtr, directBuffer.capacity());
            memoryManager.free(directBuffer);
        } else {
            long bufferPtr = MemoryUtil.memAddress(buffer);
            MemoryUtil.memCopy(bufferPtr, this.dataPtr, buffer.capacity());
        }
    }

    public void setContents(@NotNull DoubleBuffer buffer) {
        if (buffer.remaining() > this.dataSize / Double.BYTES) {
            throw new IllegalArgumentException("Cannot set contents of data container of size " + this.dataSize + " with buffer of size " + buffer.remaining());
        }
        if (!buffer.isDirect()) {
            DoubleBuffer directBuffer = memoryManager.allocDoubleBuffer(buffer.capacity());
            directBuffer.put(buffer);
            directBuffer.flip();
            long bufferPtr = MemoryUtil.memAddress(directBuffer);
            MemoryUtil.memCopy(bufferPtr, this.dataPtr, directBuffer.capacity());
            memoryManager.free(directBuffer);
        } else {
            long bufferPtr = MemoryUtil.memAddress(buffer);
            MemoryUtil.memCopy(bufferPtr, this.dataPtr, buffer.capacity());
        }
    }

    public void setContents(boolean @NotNull [] data) {
        if (data.length > this.dataSize / 8) {
            throw new IllegalArgumentException("Cannot set contents of data container of size " + this.dataSize + " with buffer of size " + data.length);
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
     * Copies the contents in the specified interval and returns it.
     *
     * @param startFlatIndex the start index of the data to copy (flat index)
     * @param endFlatIndex   the end index of the data to copy (exclusive, flat index)
     * @return the host byte buffer. Must be freed with JEmalloc.je_free()
     */
    @NotNull
    public ByteBuffer getAsDirectBuffer(long startFlatIndex, long endFlatIndex) {
        long containerSize = this.dataSize;
        if (startFlatIndex < 0 || endFlatIndex > containerSize) {
            throw new IndexOutOfBoundsException("Index out of bounds: " + startFlatIndex + " to " + endFlatIndex + " (data container length " + containerSize + ")");
        }
        if (startFlatIndex >= endFlatIndex) {
            throw new IllegalArgumentException("startFlatIndex must be less than endFlatIndex");
        }
        long nElements = endFlatIndex - startFlatIndex;
        ByteBuffer buffer = memoryManager.allocBuffer(nElements);
        long bufferPtr = MemoryUtil.memAddress(buffer);
        MemoryUtil.memCopy(this.dataPtr + startFlatIndex, bufferPtr, nElements);
        return buffer;
    }

    /**
     * @return a bytebuffer with the contents of the data container. This is not a copy, the data must not be freed.
     */
    @NotNull
    public ByteBuffer getAsDirectBuffer() {
        int dataSize = (int) min(this.dataSize, Integer.MAX_VALUE);
        return MemoryUtil.memByteBuffer(this.dataPtr, dataSize);
    }

    @NotNull
    public DataType getDataType() {
        return dataType;
    }
}
