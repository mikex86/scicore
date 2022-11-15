package me.mikex86.scicore.backend.impl.cuda;

import jcuda.Pointer;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.backend.impl.cuda.memory.CudaMemoryManager;
import me.mikex86.scicore.backend.impl.cuda.memory.CudaMemoryHandle;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.tensor.data.ITensorDataContainer;
import org.jetbrains.annotations.NotNull;

import java.nio.*;

import static jcuda.driver.JCudaDriver.*;
import static me.mikex86.scicore.backend.impl.cuda.Validator.cuCheck;

public class CudaDataContainer implements ITensorDataContainer {

    @NotNull
    private final CudaBackend backend;

    @NotNull
    private final CudaMemoryHandle deviceMemoryHandle;

    @NotNull
    private final DataType dataType;

    private final long nElements;

    public CudaDataContainer(@NotNull CudaBackend backend, @NotNull CudaMemoryManager memoryManager, long nElements, @NotNull DataType dataType) {
        this.backend = backend;
        this.deviceMemoryHandle = memoryManager.calloc(nElements, dataType);
        this.dataType = dataType;
        this.nElements = nElements;
    }

    public byte getInt8Flat(long flatIndex) {
        byte[] hostBuffer = new byte[1];
        Pointer hostPtr = Pointer.to(hostBuffer);
        cuCheck(cuMemcpyDtoH(hostPtr, deviceMemoryHandle.getDevicePointer().withByteOffset(flatIndex), 1));
        return hostBuffer[0];
    }

    public void setInt8Flat(long flatIndex, byte value) {
        byte[] hostBuffer = new byte[]{value};
        Pointer hostPtr = Pointer.to(hostBuffer);
        cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer().withByteOffset(flatIndex), hostPtr, 1));
    }

    @Override
    public short getInt16Flat(long flatIndex) {
        short[] hostBuffer = new short[1];
        Pointer hostPtr = Pointer.to(hostBuffer);
        cuCheck(cuMemcpyDtoH(hostPtr, deviceMemoryHandle.getDevicePointer().withByteOffset(flatIndex * Short.BYTES), Short.BYTES));
        return hostBuffer[0];
    }

    @Override
    public void setInt16Flat(long flatIndex, short value) {
        short[] hostBuffer = new short[]{value};
        Pointer hostPtr = Pointer.to(hostBuffer);
        cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer().withByteOffset(flatIndex * Short.BYTES), hostPtr, Short.BYTES));
    }

    @Override
    public int getInt32Flat(long flatIndex) {
        int[] hostBuffer = new int[1];
        Pointer hostPtr = Pointer.to(hostBuffer);
        cuCheck(cuMemcpyDtoH(hostPtr, deviceMemoryHandle.getDevicePointer().withByteOffset(flatIndex * Integer.BYTES), Integer.BYTES));
        return hostBuffer[0];
    }

    @Override
    public void setInt32Flat(long flatIndex, int value) {
        int[] hostBuffer = new int[]{value};
        Pointer hostPtr = Pointer.to(hostBuffer);
        cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer().withByteOffset(flatIndex * Integer.BYTES), hostPtr, Integer.BYTES));
    }

    @Override
    public long getInt64Flat(long flatIndex) {
        long[] hostBuffer = new long[1];
        Pointer hostPtr = Pointer.to(hostBuffer);
        cuCheck(cuMemcpyDtoH(hostPtr, deviceMemoryHandle.getDevicePointer().withByteOffset(flatIndex * Long.BYTES), Long.BYTES));
        return hostBuffer[0];
    }

    @Override
    public void setInt64Flat(long flatIndex, long value) {
        long[] hostBuffer = new long[]{value};
        Pointer hostPtr = Pointer.to(hostBuffer);
        cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer().withByteOffset(flatIndex * Long.BYTES), hostPtr, Long.BYTES));
    }

    @Override
    public float getFloat32Flat(long flatIndex) {
        float[] hostBuffer = new float[1];
        Pointer hostPtr = Pointer.to(hostBuffer);
        cuCheck(cuMemcpyDtoH(hostPtr, deviceMemoryHandle.getDevicePointer().withByteOffset(flatIndex * Float.BYTES), Float.BYTES));
        return hostBuffer[0];
    }

    @Override
    public void setFloat32Flat(long flatIndex, float value) {
        float[] hostBuffer = new float[]{value};
        Pointer hostPtr = Pointer.to(hostBuffer);
        cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer().withByteOffset(flatIndex * Float.BYTES), hostPtr, Float.BYTES));
    }

    @Override
    public double getFloat64Flat(long flatIndex) {
        double[] hostBuffer = new double[1];
        Pointer hostPtr = Pointer.to(hostBuffer);
        cuCheck(cuMemcpyDtoH(hostPtr, deviceMemoryHandle.getDevicePointer().withByteOffset(flatIndex * Double.BYTES), Double.BYTES));
        return hostBuffer[0];
    }

    @Override
    public void setFloat64Flat(long flatIndex, double value) {
        double[] hostBuffer = new double[]{value};
        Pointer hostPtr = Pointer.to(hostBuffer);
        cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer().withByteOffset(flatIndex * Double.BYTES), hostPtr, Double.BYTES));
    }

    @Override
    public void setBooleanFlat(long flatIndex, boolean value) {
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
        long byteIndex = flatIndex / 8;
        int bitIndex = (int) (flatIndex % 8);
        byte byteValue = getInt8Flat(byteIndex);
        return (byteValue & (1 << bitIndex)) != 0;
    }

    @Override
    public void setContents(long startIndex, @NotNull ByteBuffer hostBuffer) {
        if (hostBuffer.remaining() > deviceMemoryHandle.getSize() - startIndex) {
            throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
        }
        if (startIndex < 0 || startIndex >= deviceMemoryHandle.getSize()) {
            throw new IllegalArgumentException("Cannot set contents of buffer, flatStartIndex is out of bounds");
        }
        int size = hostBuffer.remaining();
        long byteOffset = dataType.getSizeOf(startIndex);
        if (hostBuffer.isDirect()) {
            Pointer hostPtr = Pointer.to(hostBuffer);
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer().withByteOffset(byteOffset), hostPtr, size));
        } else {
            // to direct buffer
            DirectMemoryHandle memoryHandle = backend.getDirectMemoryManager().alloc(size);
            ByteBuffer directBuffer = memoryHandle.asByteBuffer();
            directBuffer.put(hostBuffer);
            directBuffer.flip();

            Pointer hostPtr = Pointer.to(directBuffer);
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer().withByteOffset(byteOffset), hostPtr, size));

            memoryHandle.free();
        }
    }

    public void setContents(long startFlatIndex, @NotNull CudaMemoryHandle srcDevicePtr) {
        if (startFlatIndex < 0 || startFlatIndex >= deviceMemoryHandle.getSize()) {
            throw new IllegalArgumentException("Cannot set contents of buffer, startFlatIndex is out of bounds");
        }
        cuCheck(cuMemcpyDtoD(deviceMemoryHandle.getDevicePointer().withByteOffset(startFlatIndex), srcDevicePtr.getDevicePointer(), srcDevicePtr.getSize()));
    }

    @Override
    public void setContents(long startIndex, @NotNull ShortBuffer buffer) {
        if (dataType != DataType.INT16) {
            throw new IllegalArgumentException("Cannot set contents of buffer, data type is not short");
        }
        long offset = startIndex * Short.BYTES;
        if (buffer.remaining() > (deviceMemoryHandle.getSize() - offset) / Short.BYTES) {
            throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
        }
        int size = buffer.remaining() * Short.BYTES;
        if (buffer.isDirect()) {
            if (buffer.order() != ByteOrder.nativeOrder()) {
                throw new IllegalArgumentException("Direct buffer must be in native order");
            }
            Pointer hostPtr = Pointer.to(buffer);
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer().withByteOffset(offset), hostPtr, size));
        } else {
            // to direct buffer
            DirectMemoryHandle memoryHandle = backend.getDirectMemoryManager().alloc(size);
            ShortBuffer directBuffer = memoryHandle.asShortBuffer();
            directBuffer.put(buffer);
            directBuffer.flip();
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer().withByteOffset(offset), Pointer.to(directBuffer), size));
            memoryHandle.free();
        }
    }

    @Override
    public void setContents(long startIndex, @NotNull IntBuffer buffer) {
        if (dataType != DataType.INT32) {
            throw new IllegalArgumentException("Cannot set contents of buffer, data type is not int");
        }
        long offset = startIndex * Integer.BYTES;
        if (buffer.remaining() > (deviceMemoryHandle.getSize() - offset) / Integer.BYTES) {
            throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
        }
        int size = buffer.remaining() * Integer.BYTES;
        if (buffer.isDirect()) {
            if (buffer.order() != ByteOrder.nativeOrder()) {
                throw new IllegalArgumentException("Direct buffer must be in native order");
            }
            Pointer hostPtr = Pointer.to(buffer);
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer().withByteOffset(offset), hostPtr, size));
        } else {
            // to direct buffer
            DirectMemoryHandle memoryHandle = backend.getDirectMemoryManager().alloc(size);
            IntBuffer directBuffer = memoryHandle.asIntBuffer();
            directBuffer.put(buffer);
            directBuffer.flip();
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer().withByteOffset(offset), Pointer.to(directBuffer), size));
            memoryHandle.free();
        }
    }

    @Override
    public void setContents(long startIndex, @NotNull LongBuffer buffer) {
        if (dataType != DataType.INT64) {
            throw new IllegalArgumentException("Cannot set contents of buffer, data type is not long");
        }
        long offset = startIndex * Long.BYTES;
        if (buffer.remaining() > (deviceMemoryHandle.getSize() - offset) / Long.BYTES) {
            throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
        }

        int size = buffer.remaining() * Long.BYTES;
        if (buffer.isDirect()) {
            if (buffer.order() != ByteOrder.nativeOrder()) {
                throw new IllegalArgumentException("Direct buffer must be in native order");
            }
            Pointer hostPtr = Pointer.to(buffer);
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer().withByteOffset(offset), hostPtr, size));
        } else {
            // to direct buffer
            DirectMemoryHandle memoryHandle = backend.getDirectMemoryManager().alloc(size);
            LongBuffer directBuffer = memoryHandle.asLongBuffer();
            directBuffer.put(buffer);
            directBuffer.flip();
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer().withByteOffset(offset), Pointer.to(directBuffer), size));
            memoryHandle.free();
        }
    }

    @Override
    public void setContents(long startIndex, @NotNull FloatBuffer buffer) {
        if (dataType != DataType.FLOAT32) {
            throw new IllegalArgumentException("Cannot set contents of buffer, data type is not float");
        }
        long offset = startIndex * Float.BYTES;
        if (buffer.remaining() > deviceMemoryHandle.getSize() / Float.BYTES) {
            throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
        }

        int size = buffer.remaining() * Float.BYTES;
        if (buffer.isDirect()) {
            if (buffer.order() != ByteOrder.nativeOrder()) {
                throw new IllegalArgumentException("Direct buffer must be in native order");
            }
            Pointer hostPtr = Pointer.to(buffer);
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer().withByteOffset(offset), hostPtr, size));
        } else {
            // to direct buffer
            DirectMemoryHandle memoryHandle = backend.getDirectMemoryManager().alloc(size);
            FloatBuffer directBuffer = memoryHandle.asFloatBuffer();
            directBuffer.put(buffer);
            directBuffer.flip();
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer().withByteOffset(offset), Pointer.to(directBuffer), size));
            memoryHandle.free();
        }
    }

    @Override
    public void setContents(long startIndex, @NotNull DoubleBuffer buffer) {
        if (dataType != DataType.FLOAT64) {
            throw new IllegalArgumentException("Cannot set contents of buffer, data type is not double");
        }
        long offset = startIndex * Double.BYTES;
        if (buffer.remaining() > deviceMemoryHandle.getSize() / Double.BYTES) {
            throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
        }

        int size = buffer.remaining() * Double.BYTES;
        if (buffer.isDirect()) {
            if (buffer.order() != ByteOrder.nativeOrder()) {
                throw new IllegalArgumentException("Direct buffer must be in native order");
            }
            Pointer hostPtr = Pointer.to(buffer);
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer().withByteOffset(offset), hostPtr, size));
        } else {
            // to direct buffer
            DirectMemoryHandle memoryHandle = backend.getDirectMemoryManager().alloc(size);
            DoubleBuffer directBuffer = memoryHandle.asDoubleBuffer();
            directBuffer.put(buffer);
            directBuffer.flip();
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer().withByteOffset(offset), Pointer.to(directBuffer), size));
            memoryHandle.free();
        }
    }

    @Override
    public void setContents(long startIndex, boolean @NotNull [] data) {
        if (dataType != DataType.BOOLEAN) {
            throw new IllegalArgumentException("Cannot set contents of buffer, data type is not boolean");
        }
        long byteOffset = startIndex / Byte.SIZE;
        int bitOffset = Math.toIntExact(startIndex % Byte.SIZE);
        if (data.length + byteOffset > deviceMemoryHandle.getSize() * 8) {
            throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
        }
        int size = (data.length + 7) / 8; // round up to next byte
        DirectMemoryHandle memoryHandle = backend.getDirectMemoryManager().alloc(size);
        ByteBuffer buffer = memoryHandle.asByteBuffer();
        buffer.order(ByteOrder.nativeOrder());
        for (int io = 0; io < data.length; io++) {
            int i = io + bitOffset;
            int byteIndex = i / 8;
            int bitIndex = i % 8;
            byte byteValue = buffer.get(byteIndex);
            byteValue = (byte) (byteValue & ~(1 << bitIndex));
            if (data[i]) {
                byteValue = (byte) (byteValue | (1 << bitIndex));
            }
            buffer.put(byteIndex, byteValue);
        }
        if (bitOffset > 0) {
            byte prevByteValue = getInt8Flat(byteOffset);
            byte bufferValue = buffer.get(0);
            // copy first n=bitOffset bits from prevByteValue to bufferValue
            int mask = (1 << (8 - bitOffset)) - 1;
            bufferValue = (byte) (bufferValue & ~mask);
            bufferValue = (byte) (bufferValue | (prevByteValue & mask));
            buffer.put(0, bufferValue);
        }
        int nTrailingBits = (data.length + bitOffset) % 8;
        if (nTrailingBits > 0) {
            byte prevByteValue = getInt8Flat(byteOffset + size - 1);
            byte bufferValue = buffer.get(size - 1);
            // copy last n=nTrailingBits bits from prevByteValue to bufferValue
            int mask = (1 << nTrailingBits) - 1;
            bufferValue = (byte) (bufferValue & mask);
            bufferValue = (byte) (bufferValue | (prevByteValue & ~mask));
            buffer.put(size - 1, bufferValue);
        }
        buffer.flip();
        Pointer hostPtr = Pointer.to(buffer);
        cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer().withByteOffset(byteOffset), hostPtr, size));
        memoryHandle.free();
    }

    /**
     * Copies the contents in the specified interval from the cuda device to host memory and returns it.
     *
     * @param startFlatIndex the start index of the data to copy (flat index)
     * @param endFlatIndex   the end index of the data to copy (exclusive, flat index)
     * @return the host byte buffer. Must be freed via the direct memory manager.
     */
    @NotNull
    @Override
    public DirectMemoryHandle getAsDirectBuffer(long startFlatIndex, long endFlatIndex) {
        long containerSize = deviceMemoryHandle.getSize();
        if (startFlatIndex < 0 || endFlatIndex > containerSize) {
            throw new IndexOutOfBoundsException("Index out of bounds: " + startFlatIndex + " to " + endFlatIndex + " (data container length " + containerSize + ")");
        }
        if (startFlatIndex >= endFlatIndex) {
            throw new IllegalArgumentException("startFlatIndex must be less than endFlatIndex");
        }
        long nElements = endFlatIndex - startFlatIndex;
        DirectMemoryHandle memoryHandle = backend.getDirectMemoryManager().alloc(nElements);
        ByteBuffer buffer = memoryHandle.asByteBuffer();
        Pointer hostPtr = Pointer.to(buffer);
        cuCheck(cuMemcpyDtoH(hostPtr, deviceMemoryHandle.getDevicePointer().withByteOffset(dataType.getSizeOf(startFlatIndex)), (nElements * dataType.getBits() + 7) / 8));
        buffer.flip();
        return memoryHandle;
    }

    @Override
    public @NotNull DirectMemoryHandle getAsDirectBuffer() {
        return getAsDirectBuffer(0, deviceMemoryHandle.getSize());
    }

    @Override
    public @NotNull DataType getDataType() {
        return dataType;
    }

    @Override
    public long getDataSize() {
        return deviceMemoryHandle.getSize();
    }

    @Override
    public long getNumberOfElements() {
        return this.nElements;
    }

    @NotNull
    public CudaMemoryHandle getDeviceMemoryHandle() {
        return deviceMemoryHandle;
    }


    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, byte value) {
        throw new UnsupportedOperationException("TODO: implement");
    }


    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, short value) {
        throw new UnsupportedOperationException("TODO: implement");
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, int value) {
        throw new UnsupportedOperationException("TODO: implement");
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, long value) {
        throw new UnsupportedOperationException("TODO: implement");
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, float value) {
        throw new UnsupportedOperationException("TODO: implement");
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, double value) {
        throw new UnsupportedOperationException("TODO: implement");
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, boolean value) {
        throw new UnsupportedOperationException("TODO: implement");
    }

    @Override
    public void fill(boolean value) {
        byte byteValue = value ? (byte) 0xFF : (byte) 0x00;
        fill(byteValue);
    }

    @Override
    public void dispose() {
        deviceMemoryHandle.free();
    }
}
