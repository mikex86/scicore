package me.mikex86.scicore.backend.impl.genericcpu;

import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.memory.DirectMemoryManager;
import me.mikex86.scicore.tensor.data.ITensorDataContainer;
import org.jetbrains.annotations.NotNull;
import org.lwjgl.system.MemoryUtil;

import java.nio.*;

import static java.lang.Math.min;

public class GenCPUTensorDataContainer implements ITensorDataContainer {

    @NotNull
    private final DirectMemoryHandle memoryHandle;

    private final long dataSize;

    @NotNull
    private final DataType dataType;

    @NotNull
    private final DirectMemoryManager memoryManager;

    private boolean disposed = false;

    private final long nElements;

    public GenCPUTensorDataContainer(@NotNull DirectMemoryManager memoryManager, long nElements, @NotNull DataType dataType) {
        this.memoryManager = memoryManager;
        long nBytes = dataType.getSizeOf(nElements);
        this.memoryHandle = memoryManager.calloc(nBytes);
        this.dataSize = nBytes;
        this.dataType = dataType;
        this.nElements = nElements;
    }

    private void checkDisposed() {
        if (disposed) {
            throw new IllegalStateException("GenCpuTensorDataContainer has already been disposed");
        }
    }

    @Override
    public byte getInt8Flat(long flatIndex) {
        checkDisposed();
        long finalPtr = memoryHandle.getNativePtr() + flatIndex;
        return MemoryUtil.memGetByte(finalPtr);
    }

    @Override
    public void setInt8Flat(long flatIndex, byte value) {
        checkDisposed();
        long finalPtr = memoryHandle.getNativePtr() + flatIndex;
        MemoryUtil.memPutByte(finalPtr, value);
    }

    @Override
    public short getInt16Flat(long flatIndex) {
        checkDisposed();
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 2;
        return MemoryUtil.memGetShort(finalPtr);
    }

    @Override
    public void setInt16Flat(long flatIndex, short value) {
        checkDisposed();
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 2;
        MemoryUtil.memPutShort(finalPtr, value);
    }

    @Override
    public int getInt32Flat(long flatIndex) {
        checkDisposed();
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 4;
        return MemoryUtil.memGetInt(finalPtr);
    }

    @Override
    public void setInt32Flat(long flatIndex, int value) {
        checkDisposed();
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 4;
        MemoryUtil.memPutInt(finalPtr, value);
    }

    @Override
    public long getInt64Flat(long flatIndex) {
        checkDisposed();
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 8;
        return MemoryUtil.memGetLong(finalPtr);
    }

    @Override
    public void setInt64Flat(long flatIndex, long value) {
        checkDisposed();
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 8;
        MemoryUtil.memPutLong(finalPtr, value);
    }

    @Override
    public float getFloat32Flat(long flatIndex) {
        checkDisposed();
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 4;
        return MemoryUtil.memGetFloat(finalPtr);
    }

    @Override
    public void setFloat32Flat(long flatIndex, float value) {
        checkDisposed();
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 4;
        MemoryUtil.memPutFloat(finalPtr, value);
    }

    @Override
    public double getFloat64Flat(long flatIndex) {
        checkDisposed();
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 8;
        return MemoryUtil.memGetDouble(finalPtr);
    }

    @Override
    public void setFloat64Flat(long flatIndex, double value) {
        checkDisposed();
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 8;
        MemoryUtil.memPutDouble(finalPtr, value);
    }

    @Override
    public void setBooleanFlat(long flatIndex, boolean value) {
        checkDisposed();
        long byteIndex = flatIndex / 8;
        int bitIndex = (int) (flatIndex % 8);
        byte byteValue = getInt8Flat(byteIndex);
        byteValue = (byte) (byteValue & ~(1 << bitIndex));
        if (value) {
            byteValue = (byte) (byteValue | (1 << bitIndex));
        }
        setInt8Flat(byteIndex, byteValue);
    }

    @Override
    public boolean getBooleanFlat(long flatIndex) {
        checkDisposed();
        long byteIndex = flatIndex / 8;
        int bitIndex = (int) (flatIndex % 8);
        byte byteValue = getInt8Flat(byteIndex);
        return (byteValue & (1 << bitIndex)) != 0;
    }

    @Override
    public void setContents(long startIndex, @NotNull ByteBuffer buffer) {
        checkDisposed();
        if (buffer.remaining() > this.dataSize - startIndex) {
            throw new IllegalArgumentException("Cannot set contents of data container of size " + this.dataSize + " with buffer of size " + buffer.remaining());
        }
        long byteOffset = dataType.getSizeOf(startIndex);
        if (!buffer.isDirect()) {
            DirectMemoryHandle memoryHandle = memoryManager.alloc(buffer.capacity());
            ByteBuffer directBuffer = memoryHandle.asByteBuffer();
            directBuffer.put(buffer);
            directBuffer.flip();
            long bufferPtr = memoryHandle.getNativePtr();
            MemoryUtil.memCopy(bufferPtr, this.memoryHandle.getNativePtr() + byteOffset, directBuffer.capacity());
            memoryHandle.free();
        } else {
            long bufferPtr = MemoryUtil.memAddress(buffer);
            MemoryUtil.memCopy(bufferPtr, this.memoryHandle.getNativePtr() + byteOffset, buffer.capacity());
        }
    }

    @Override
    public void setContents(long startIndex, @NotNull ShortBuffer buffer) {
        if (dataType != DataType.INT16) {
            throw new IllegalArgumentException("Cannot set contents of data container of type " + dataType + " with buffer of type " + DataType.INT16);
        }
        checkDisposed();
        long offset = startIndex * Short.BYTES;
        if (buffer.remaining() > (this.dataSize - offset) / Short.BYTES) {
            throw new IllegalArgumentException("Cannot set contents of data container of size " + this.dataSize + " with buffer of size " + buffer.remaining());
        }
        long nBytes = (long) buffer.capacity() * Short.BYTES;
        if (!buffer.isDirect()) {
            DirectMemoryHandle memoryHandle = memoryManager.alloc(nBytes);
            ShortBuffer directBuffer = memoryHandle.asShortBuffer();
            directBuffer.put(buffer);
            directBuffer.flip();
            long bufferPtr = memoryHandle.getNativePtr();
            MemoryUtil.memCopy(bufferPtr, this.memoryHandle.getNativePtr() + offset, nBytes);
            memoryHandle.free();
        } else {
            long bufferPtr = MemoryUtil.memAddress(buffer);
            MemoryUtil.memCopy(bufferPtr, this.memoryHandle.getNativePtr() + offset, nBytes);
        }
    }

    @Override
    public void setContents(long startIndex, @NotNull IntBuffer buffer) {
        if (dataType != DataType.INT32) {
            throw new IllegalArgumentException("Cannot set contents of data container of type " + dataType + " with buffer of type " + DataType.INT32);
        }
        checkDisposed();
        long offset = startIndex * Integer.BYTES;
        if (buffer.remaining() > (this.dataSize - offset) / Integer.BYTES) {
            throw new IllegalArgumentException("Cannot set contents of data container of size " + this.dataSize + " with buffer of size " + buffer.remaining());
        }
        long nBytes = (long) buffer.capacity() * Integer.BYTES;
        if (!buffer.isDirect()) {
            DirectMemoryHandle memoryHandle = memoryManager.alloc(nBytes);
            IntBuffer directBuffer = memoryHandle.asIntBuffer();
            directBuffer.put(buffer);
            directBuffer.flip();
            long bufferPtr = memoryHandle.getNativePtr();
            MemoryUtil.memCopy(bufferPtr, this.memoryHandle.getNativePtr() + offset, nBytes);
            memoryHandle.free();
        } else {
            long bufferPtr = MemoryUtil.memAddress(buffer);
            MemoryUtil.memCopy(bufferPtr, this.memoryHandle.getNativePtr() + offset, nBytes);
        }
    }

    @Override
    public void setContents(long startIndex, @NotNull LongBuffer buffer) {
        if (dataType != DataType.INT64) {
            throw new IllegalArgumentException("Cannot set contents of data container of type " + dataType + " with buffer of type " + DataType.INT64);
        }
        checkDisposed();
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
    public void setContents(long startIndex, @NotNull FloatBuffer buffer) {
        if (dataType != DataType.FLOAT32) {
            throw new IllegalArgumentException("Cannot set contents of data container of type " + dataType + " with buffer of type " + DataType.FLOAT32);
        }
        checkDisposed();
        long offset = startIndex * Float.BYTES;
        if (buffer.remaining() > (this.dataSize - offset) / Float.BYTES) {
            throw new IllegalArgumentException("Cannot set contents of data container of size " + this.dataSize + " with buffer of size " + buffer.remaining());
        }
        long nBytes = (long) buffer.capacity() * Float.BYTES;
        if (!buffer.isDirect()) {
            DirectMemoryHandle memoryHandle = memoryManager.alloc(nBytes);
            FloatBuffer directBuffer = memoryHandle.asFloatBuffer();
            directBuffer.put(buffer);
            directBuffer.flip();
            long bufferPtr = memoryHandle.getNativePtr();
            MemoryUtil.memCopy(bufferPtr, this.memoryHandle.getNativePtr() + offset, nBytes);
            memoryHandle.free();
        } else {
            long bufferPtr = MemoryUtil.memAddress(buffer);
            MemoryUtil.memCopy(bufferPtr, this.memoryHandle.getNativePtr() + offset, nBytes);
        }
    }

    @Override
    public void setContents(long startIndex, @NotNull DoubleBuffer buffer) {
        if (dataType != DataType.FLOAT64) {
            throw new IllegalArgumentException("Cannot set contents of data container of type " + dataType + " with buffer of type " + DataType.FLOAT64);
        }
        checkDisposed();
        long offset = startIndex * Double.BYTES;
        if (buffer.remaining() > (this.dataSize - offset) / Double.BYTES) {
            throw new IllegalArgumentException("Cannot set contents of data container of size " + this.dataSize + " with buffer of size " + buffer.remaining());
        }
        long nBytes = (long) buffer.capacity() * Double.BYTES;
        if (!buffer.isDirect()) {
            DirectMemoryHandle memoryHandle = memoryManager.alloc(nBytes);
            DoubleBuffer directBuffer = memoryHandle.asDoubleBuffer();
            directBuffer.put(buffer);
            directBuffer.flip();
            long bufferPtr = memoryHandle.getNativePtr();
            MemoryUtil.memCopy(bufferPtr, this.memoryHandle.getNativePtr() + offset, nBytes);
            memoryHandle.free();
        } else {
            long bufferPtr = MemoryUtil.memAddress(buffer);
            MemoryUtil.memCopy(bufferPtr, this.memoryHandle.getNativePtr() + offset, nBytes);
        }
    }

    @Override
    public void setContents(long startIndex, boolean @NotNull [] data) {
        if (dataType != DataType.BOOLEAN) {
            throw new IllegalArgumentException("Cannot set contents of data container of type " + dataType + " with array of type " + DataType.BOOLEAN);
        }
        checkDisposed();
        long byteOffset = startIndex / Byte.SIZE;
        if (data.length + startIndex > this.dataSize * 8) {
            throw new IllegalArgumentException("Cannot set contents of data container of size " + this.dataSize * 8 + " bits with buffer of size " + data.length + " bits");
        }
        for (int i = 0; i < data.length; i++) {
            int byteIndex = i / 8;
            int bitIndex = i % 8;
            byte byteValue = getInt8Flat(byteOffset + byteIndex);
            if (data[i]) {
                byteValue = (byte) (byteValue | (1 << bitIndex));
            } else {
                byteValue = (byte) (byteValue & ~(1 << bitIndex));
            }
            setInt8Flat(byteOffset + byteIndex, byteValue);
        }
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, byte value) {
        checkDisposed();
        if (startFlatIndex < 0 || startFlatIndex >= this.nElements) {
            throw new IllegalArgumentException("Start flat index " + startFlatIndex + " is out of bounds");
        }
        if (endFlatIndex < 0 || endFlatIndex > this.nElements) {
            throw new IllegalArgumentException("End flat index " + endFlatIndex + " is out of bounds");
        }
        if (endFlatIndex < startFlatIndex) {
            throw new IllegalArgumentException("End flat index " + endFlatIndex + " is less than start flat index " + startFlatIndex);
        }
        long nElementsInRegion = endFlatIndex - startFlatIndex;
        long nBytes = dataType.getSizeOf(nElementsInRegion);
        switch (dataType) {
            case INT8 -> MemoryUtil.memSet(this.memoryHandle.getNativePtr() + startFlatIndex, value, nBytes);
            case INT16 -> {
                ShortBuffer shortBuffer = this.memoryHandle.asShortBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    shortBuffer.put(Math.toIntExact(startFlatIndex + i), value);
                }
            }
            case INT32 -> {
                IntBuffer intBuffer = this.memoryHandle.asIntBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    intBuffer.put(Math.toIntExact(startFlatIndex + i), value);
                }
            }
            case INT64 -> {
                LongBuffer longBuffer = this.memoryHandle.asLongBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    longBuffer.put(Math.toIntExact(startFlatIndex + i), value);
                }
            }
            case FLOAT32 -> {
                FloatBuffer floatBuffer = this.memoryHandle.asFloatBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    floatBuffer.put(Math.toIntExact(startFlatIndex + i), value);
                }
            }
            case FLOAT64 -> {
                DoubleBuffer doubleBuffer = this.memoryHandle.asDoubleBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    doubleBuffer.put(Math.toIntExact(startFlatIndex + i), value);
                }
            }
            case BOOLEAN -> {
                for (int i = 0; i < nElementsInRegion; i++) {
                    setBooleanFlat(startFlatIndex + i, value != 0);
                }
            }
        }
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, short value) {
        checkDisposed();
        if (startFlatIndex < 0 || startFlatIndex >= this.nElements) {
            throw new IllegalArgumentException("Start flat index " + startFlatIndex + " is out of bounds");
        }
        if (endFlatIndex < 0 || endFlatIndex > this.nElements) {
            throw new IllegalArgumentException("End flat index " + endFlatIndex + " is out of bounds");
        }
        if (endFlatIndex < startFlatIndex) {
            throw new IllegalArgumentException("End flat index " + endFlatIndex + " is less than start flat index " + startFlatIndex);
        }
        long nElementsInRegion = endFlatIndex - startFlatIndex;
        long nBytes = dataType.getSizeOf(nElementsInRegion);
        switch (dataType) {
            case INT8 -> MemoryUtil.memSet(this.memoryHandle.getNativePtr() + startFlatIndex, value, nBytes);
            case INT16 -> {
                ShortBuffer shortBuffer = this.memoryHandle.asShortBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    shortBuffer.put(Math.toIntExact(startFlatIndex + i), value);
                }
            }
            case INT32 -> {
                IntBuffer intBuffer = this.memoryHandle.asIntBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    intBuffer.put(Math.toIntExact(startFlatIndex + i), value);
                }
            }
            case INT64 -> {
                LongBuffer longBuffer = this.memoryHandle.asLongBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    longBuffer.put(Math.toIntExact(startFlatIndex + i), value);
                }
            }
            case FLOAT32 -> {
                FloatBuffer floatBuffer = this.memoryHandle.asFloatBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    floatBuffer.put(Math.toIntExact(startFlatIndex + i), value);
                }
            }
            case FLOAT64 -> {
                DoubleBuffer doubleBuffer = this.memoryHandle.asDoubleBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    doubleBuffer.put(Math.toIntExact(startFlatIndex + i), value);
                }
            }
            case BOOLEAN -> {
                for (int i = 0; i < nElementsInRegion; i++) {
                    setBooleanFlat(startFlatIndex + i, value != 0);
                }
            }
        }
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, int value) {
        checkDisposed();
        if (startFlatIndex < 0 || startFlatIndex >= this.nElements) {
            throw new IllegalArgumentException("Start flat index " + startFlatIndex + " is out of bounds");
        }
        if (endFlatIndex < 0 || endFlatIndex > this.nElements) {
            throw new IllegalArgumentException("End flat index " + endFlatIndex + " is out of bounds");
        }
        if (endFlatIndex < startFlatIndex) {
            throw new IllegalArgumentException("End flat index " + endFlatIndex + " is less than start flat index " + startFlatIndex);
        }
        long nElementsInRegion = endFlatIndex - startFlatIndex;
        long nBytes = dataType.getSizeOf(nElementsInRegion);
        switch (dataType) {
            case INT8 -> MemoryUtil.memSet(this.memoryHandle.getNativePtr() + startFlatIndex, value, nBytes);
            case INT16 -> {
                ShortBuffer shortBuffer = this.memoryHandle.asShortBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    shortBuffer.put(Math.toIntExact(startFlatIndex + i), (short) value);
                }
            }
            case INT32 -> {
                IntBuffer intBuffer = this.memoryHandle.asIntBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    intBuffer.put(Math.toIntExact(startFlatIndex + i), value);
                }
            }
            case INT64 -> {
                LongBuffer longBuffer = this.memoryHandle.asLongBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    longBuffer.put(Math.toIntExact(startFlatIndex + i), value);
                }
            }
            case FLOAT32 -> {
                FloatBuffer floatBuffer = this.memoryHandle.asFloatBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    floatBuffer.put(Math.toIntExact(startFlatIndex + i), value);
                }
            }
            case FLOAT64 -> {
                DoubleBuffer doubleBuffer = this.memoryHandle.asDoubleBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    doubleBuffer.put(Math.toIntExact(startFlatIndex + i), value);
                }
            }
            case BOOLEAN -> {
                for (int i = 0; i < nElementsInRegion; i++) {
                    setBooleanFlat(startFlatIndex + i, value != 0);
                }
            }
        }
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, long value) {
        checkDisposed();
        if (startFlatIndex < 0 || startFlatIndex >= this.nElements) {
            throw new IllegalArgumentException("Start flat index " + startFlatIndex + " is out of bounds");
        }
        if (endFlatIndex < 0 || endFlatIndex > this.nElements) {
            throw new IllegalArgumentException("End flat index " + endFlatIndex + " is out of bounds");
        }
        if (endFlatIndex < startFlatIndex) {
            throw new IllegalArgumentException("End flat index " + endFlatIndex + " is less than start flat index " + startFlatIndex);
        }
        long nElementsInRegion = endFlatIndex - startFlatIndex;
        long nBytes = dataType.getSizeOf(nElementsInRegion);
        switch (dataType) {
            case INT8 -> MemoryUtil.memSet(this.memoryHandle.getNativePtr() + startFlatIndex, (int) value, nBytes);
            case INT16 -> {
                ShortBuffer shortBuffer = this.memoryHandle.asShortBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    shortBuffer.put(Math.toIntExact(startFlatIndex + i), (short) value);
                }
            }
            case INT32 -> {
                IntBuffer intBuffer = this.memoryHandle.asIntBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    intBuffer.put(Math.toIntExact(startFlatIndex + i), (int) value);
                }
            }
            case INT64 -> {
                LongBuffer longBuffer = this.memoryHandle.asLongBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    longBuffer.put(Math.toIntExact(startFlatIndex + i), value);
                }
            }
            case FLOAT32 -> {
                FloatBuffer floatBuffer = this.memoryHandle.asFloatBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    floatBuffer.put(Math.toIntExact(startFlatIndex + i), value);
                }
            }
            case FLOAT64 -> {
                DoubleBuffer doubleBuffer = this.memoryHandle.asDoubleBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    doubleBuffer.put(Math.toIntExact(startFlatIndex + i), value);
                }
            }
            case BOOLEAN -> {
                for (int i = 0; i < nElementsInRegion; i++) {
                    setBooleanFlat(startFlatIndex + i, value != 0);
                }
            }
        }
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, float value) {
        checkDisposed();
        if (startFlatIndex < 0 || startFlatIndex >= this.nElements) {
            throw new IllegalArgumentException("Start flat index " + startFlatIndex + " is out of bounds");
        }
        if (endFlatIndex < 0 || endFlatIndex > this.nElements) {
            throw new IllegalArgumentException("End flat index " + endFlatIndex + " is out of bounds");
        }
        if (endFlatIndex < startFlatIndex) {
            throw new IllegalArgumentException("End flat index " + endFlatIndex + " is less than start flat index " + startFlatIndex);
        }
        long nElementsInRegion = endFlatIndex - startFlatIndex;
        long nBytes = dataType.getSizeOf(nElementsInRegion);
        switch (dataType) {
            case INT8 -> MemoryUtil.memSet(this.memoryHandle.getNativePtr() + startFlatIndex, (int) value, nBytes);
            case INT16 -> {
                ShortBuffer shortBuffer = this.memoryHandle.asShortBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    shortBuffer.put(Math.toIntExact(startFlatIndex + i), (short) value);
                }
            }
            case INT32 -> {
                IntBuffer intBuffer = this.memoryHandle.asIntBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    intBuffer.put(Math.toIntExact(startFlatIndex + i), (int) value);
                }
            }
            case INT64 -> {
                LongBuffer longBuffer = this.memoryHandle.asLongBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    longBuffer.put(Math.toIntExact(startFlatIndex + i), (long) value);
                }
            }
            case FLOAT32 -> {
                FloatBuffer floatBuffer = this.memoryHandle.asFloatBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    floatBuffer.put(Math.toIntExact(startFlatIndex + i), value);
                }
            }
            case FLOAT64 -> {
                DoubleBuffer doubleBuffer = this.memoryHandle.asDoubleBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    doubleBuffer.put(Math.toIntExact(startFlatIndex + i), value);
                }
            }
            case BOOLEAN -> {
                for (int i = 0; i < nElementsInRegion; i++) {
                    setBooleanFlat(startFlatIndex + i, value != 0);
                }
            }
        }
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, double value) {
        checkDisposed();
        if (startFlatIndex < 0 || startFlatIndex >= this.nElements) {
            throw new IllegalArgumentException("Start flat index " + startFlatIndex + " is out of bounds");
        }
        if (endFlatIndex < 0 || endFlatIndex > this.nElements) {
            throw new IllegalArgumentException("End flat index " + endFlatIndex + " is out of bounds");
        }
        if (endFlatIndex < startFlatIndex) {
            throw new IllegalArgumentException("End flat index " + endFlatIndex + " is less than start flat index " + startFlatIndex);
        }
        long nElementsInRegion = endFlatIndex - startFlatIndex;
        long nBytes = dataType.getSizeOf(nElementsInRegion);
        switch (dataType) {
            case INT8 -> MemoryUtil.memSet(this.memoryHandle.getNativePtr() + startFlatIndex, (int) value, nBytes);
            case INT16 -> {
                ShortBuffer shortBuffer = this.memoryHandle.asShortBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    shortBuffer.put(Math.toIntExact(startFlatIndex + i), (short) value);
                }
            }
            case INT32 -> {
                IntBuffer intBuffer = this.memoryHandle.asIntBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    intBuffer.put(Math.toIntExact(startFlatIndex + i), (int) value);
                }
            }
            case INT64 -> {
                LongBuffer longBuffer = this.memoryHandle.asLongBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    longBuffer.put(Math.toIntExact(startFlatIndex + i), (long) value);
                }
            }
            case FLOAT32 -> {
                FloatBuffer floatBuffer = this.memoryHandle.asFloatBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    floatBuffer.put(Math.toIntExact(startFlatIndex + i), (float) value);
                }
            }
            case FLOAT64 -> {
                DoubleBuffer doubleBuffer = this.memoryHandle.asDoubleBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    doubleBuffer.put(Math.toIntExact(startFlatIndex + i), value);
                }
            }
            case BOOLEAN -> {
                for (int i = 0; i < nElementsInRegion; i++) {
                    setBooleanFlat(startFlatIndex + i, value != 0);
                }
            }
        }
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, boolean value) {
        checkDisposed();
        if (startFlatIndex < 0 || startFlatIndex >= this.nElements) {
            throw new IllegalArgumentException("Start flat index " + startFlatIndex + " is out of bounds");
        }
        if (endFlatIndex < 0 || endFlatIndex > this.nElements) {
            throw new IllegalArgumentException("End flat index " + endFlatIndex + " is out of bounds");
        }
        if (endFlatIndex < startFlatIndex) {
            throw new IllegalArgumentException("End flat index " + endFlatIndex + " is less than start flat index " + startFlatIndex);
        }
        long nElementsInRegion = endFlatIndex - startFlatIndex;
        long nBytes = dataType.getSizeOf(nElementsInRegion);
        switch (dataType) {
            case INT8 -> MemoryUtil.memSet(this.memoryHandle.getNativePtr() + startFlatIndex, value ? 1 : 0, nBytes);
            case INT16 -> {
                ShortBuffer shortBuffer = this.memoryHandle.asShortBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    shortBuffer.put(Math.toIntExact(startFlatIndex + i), (short) (value ? 1 : 0));
                }
            }
            case INT32 -> {
                IntBuffer intBuffer = this.memoryHandle.asIntBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    intBuffer.put(Math.toIntExact(startFlatIndex + i), value ? 1 : 0);
                }
            }
            case INT64 -> {
                LongBuffer longBuffer = this.memoryHandle.asLongBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    longBuffer.put(Math.toIntExact(startFlatIndex + i), value ? 1 : 0);
                }
            }
            case FLOAT32 -> {
                FloatBuffer floatBuffer = this.memoryHandle.asFloatBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    floatBuffer.put(Math.toIntExact(startFlatIndex + i), value ? 1 : 0);
                }
            }
            case FLOAT64 -> {
                DoubleBuffer doubleBuffer = this.memoryHandle.asDoubleBuffer();
                for (int i = 0; i < nElementsInRegion; i++) {
                    doubleBuffer.put(Math.toIntExact(startFlatIndex + i), value ? 1 : 0);
                }
            }
            case BOOLEAN -> {
                for (int i = 0; i < nElementsInRegion; i++) {
                    setBooleanFlat(startFlatIndex + i, value);
                }
            }
        }
    }

    @Override
    public long getNumberOfElements() {
        return this.dataSize;
    }

    @Override
    @NotNull
    public DirectMemoryHandle getAsDirectBuffer(long startFlatIndex, long endFlatIndex) {
        checkDisposed();
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
        checkDisposed();
        return this.memoryHandle.createReference();
    }

    @Override
    @NotNull
    public DataType getDataType() {
        return dataType;
    }

    @NotNull
    public DirectMemoryHandle getMemoryHandle() {
        checkDisposed();
        return memoryHandle;
    }

    @Override
    public long getDataSize() {
        return dataSize;
    }

    @Override
    public void dispose() {
        if (this.disposed) {
            throw new IllegalStateException("Data container already disposed");
        }
        this.memoryHandle.free();
        this.disposed = true;
    }
}