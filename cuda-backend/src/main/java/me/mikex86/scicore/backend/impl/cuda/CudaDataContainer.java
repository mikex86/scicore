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

    public CudaDataContainer(@NotNull CudaBackend backend, @NotNull CudaMemoryManager memoryManager, long nElements, @NotNull DataType dataType) {
        this.backend = backend;
        this.deviceMemoryHandle = memoryManager.calloc(nElements, dataType);
        this.dataType = dataType;
    }

    public byte getInt8Flat(long flatIndex) {
        byte[] hostBuffer = new byte[1];
        Pointer hostPtr = Pointer.to(hostBuffer);
        cuCheck(cuMemcpyDtoH(hostPtr, deviceMemoryHandle.getDevicePointer().withByteOffset(flatIndex), 1));
        return hostBuffer[0];
    }

    public void getInt8Flat(byte value, long flatIndex) {
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
    public void setInt16Flat(short value, long flatIndex) {
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
    public void setInt32Flat(int value, long flatIndex) {
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
    public void setInt64Flat(long value, long flatIndex) {
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
    public void setFloat32Flat(float value, long flatIndex) {
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
    public void setFloat64Flat(double value, long flatIndex) {
        double[] hostBuffer = new double[]{value};
        Pointer hostPtr = Pointer.to(hostBuffer);
        cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer().withByteOffset(flatIndex * Double.BYTES), hostPtr, Double.BYTES));
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


    public void setContents(@NotNull ByteBuffer buffer) {
        if (buffer.remaining() > deviceMemoryHandle.getSize()) {
            throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
        }
        int size = buffer.remaining();
        if (buffer.isDirect()) {
            Pointer hostPtr = Pointer.to(buffer);
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer(), hostPtr, size));
        } else {
            // to direct buffer
            DirectMemoryHandle memoryHandle = backend.getDirectMemoryManager().alloc(size);
            ByteBuffer directBuffer = memoryHandle.asByteBuffer();
            directBuffer.put(buffer);
            directBuffer.flip();

            Pointer hostPtr = Pointer.to(buffer);
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer(), hostPtr, size));

            memoryHandle.free();
        }
    }

    public void setContents(@NotNull ByteBuffer hostBuffer, long flatStartIndex) {
        if (hostBuffer.remaining() > deviceMemoryHandle.getSize() - flatStartIndex) {
            throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
        }
        if (flatStartIndex < 0 || flatStartIndex >= deviceMemoryHandle.getSize()) {
            throw new IllegalArgumentException("Cannot set contents of buffer, flatStartIndex is out of bounds");
        }
        int size = hostBuffer.remaining();
        if (hostBuffer.isDirect()) {
            Pointer hostPtr = Pointer.to(hostBuffer);
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer().withByteOffset(flatStartIndex), hostPtr, size));
        } else {
            // to direct buffer
            DirectMemoryHandle memoryHandle = backend.getDirectMemoryManager().alloc(size);
            ByteBuffer directBuffer = memoryHandle.asByteBuffer();
            directBuffer.put(hostBuffer);
            directBuffer.flip();

            Pointer hostPtr = Pointer.to(directBuffer);
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer().withByteOffset(flatStartIndex), hostPtr, size));

            memoryHandle.free();
        }
    }

    public void setContents(@NotNull CudaMemoryHandle srcDevicePtr, long startFlatIndex) {
        if (startFlatIndex < 0 || startFlatIndex >= deviceMemoryHandle.getSize()) {
            throw new IllegalArgumentException("Cannot set contents of buffer, startFlatIndex is out of bounds");
        }
        cuCheck(cuMemcpyDtoD(deviceMemoryHandle.getDevicePointer().withByteOffset(startFlatIndex), srcDevicePtr.getDevicePointer(), srcDevicePtr.getSize()));
    }

    public void setContents(@NotNull ShortBuffer buffer) {
        if (buffer.remaining() > deviceMemoryHandle.getSize() / Short.BYTES) {
            throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
        }

        int size = buffer.remaining() * Short.BYTES;
        if (buffer.isDirect()) {
            if (buffer.order() != ByteOrder.nativeOrder()) {
                throw new IllegalArgumentException("Direct buffer must be in native order");
            }
            Pointer hostPtr = Pointer.to(buffer);
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer(), hostPtr, size));
        } else {
            // to direct buffer
            DirectMemoryHandle memoryHandle = backend.getDirectMemoryManager().alloc(size);
            ShortBuffer directBuffer = memoryHandle.asShortBuffer();
            directBuffer.put(buffer);
            directBuffer.flip();
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer(), Pointer.to(directBuffer), size));
            memoryHandle.free();
        }
    }

    public void setContents(@NotNull IntBuffer buffer) {
        if (buffer.remaining() > deviceMemoryHandle.getSize() / Integer.BYTES) {
            throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
        }

        int size = buffer.remaining() * Integer.BYTES;
        if (buffer.isDirect()) {
            if (buffer.order() != ByteOrder.nativeOrder()) {
                throw new IllegalArgumentException("Direct buffer must be in native order");
            }
            Pointer hostPtr = Pointer.to(buffer);
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer(), hostPtr, size));
        } else {
            // to direct buffer
            DirectMemoryHandle memoryHandle = backend.getDirectMemoryManager().alloc(size);
            IntBuffer directBuffer = memoryHandle.asIntBuffer();
            directBuffer.put(buffer);
            directBuffer.flip();
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer(), Pointer.to(directBuffer), size));
            memoryHandle.free();
        }
    }

    public void setContents(@NotNull LongBuffer buffer) {
        if (buffer.remaining() > deviceMemoryHandle.getSize() / Long.BYTES) {
            throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
        }

        int size = buffer.remaining() * Long.BYTES;
        if (buffer.isDirect()) {
            if (buffer.order() != ByteOrder.nativeOrder()) {
                throw new IllegalArgumentException("Direct buffer must be in native order");
            }
            Pointer hostPtr = Pointer.to(buffer);
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer(), hostPtr, size));
        } else {
            // to direct buffer
            DirectMemoryHandle memoryHandle = backend.getDirectMemoryManager().alloc(size);
            LongBuffer directBuffer = memoryHandle.asLongBuffer();
            directBuffer.put(buffer);
            directBuffer.flip();
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer(), Pointer.to(directBuffer), size));
            memoryHandle.free();
        }
    }

    public void setContents(@NotNull FloatBuffer buffer) {
        if (buffer.remaining() > deviceMemoryHandle.getSize() / Float.BYTES) {
            throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
        }

        int size = buffer.remaining() * Float.BYTES;
        if (buffer.isDirect()) {
            if (buffer.order() != ByteOrder.nativeOrder()) {
                throw new IllegalArgumentException("Direct buffer must be in native order");
            }
            Pointer hostPtr = Pointer.to(buffer);
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer(), hostPtr, size));
        } else {
            // to direct buffer
            DirectMemoryHandle memoryHandle = backend.getDirectMemoryManager().alloc(size);
            FloatBuffer directBuffer = memoryHandle.asFloatBuffer();
            directBuffer.put(buffer);
            directBuffer.flip();
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer(), Pointer.to(directBuffer), size));
            memoryHandle.free();
        }
    }

    public void setContents(@NotNull DoubleBuffer buffer) {
        if (buffer.remaining() > deviceMemoryHandle.getSize() / Double.BYTES) {
            throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
        }

        int size = buffer.remaining() * Double.BYTES;
        if (buffer.isDirect()) {
            if (buffer.order() != ByteOrder.nativeOrder()) {
                throw new IllegalArgumentException("Direct buffer must be in native order");
            }
            Pointer hostPtr = Pointer.to(buffer);
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer(), hostPtr, size));
        } else {
            // to direct buffer
            DirectMemoryHandle memoryHandle = backend.getDirectMemoryManager().alloc(size);
            DoubleBuffer directBuffer = memoryHandle.asDoubleBuffer();
            directBuffer.put(buffer);
            directBuffer.flip();
            cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer(), Pointer.to(directBuffer), size));
            memoryHandle.free();
        }
    }

    public void setContents(boolean @NotNull [] data) {
        if (data.length > deviceMemoryHandle.getSize() * 8) {
            throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
        }
        int size = (data.length + 7) / 8; // round up to next byte
        DirectMemoryHandle memoryHandle = backend.getDirectMemoryManager().alloc(size);
        ByteBuffer buffer = memoryHandle.asByteBuffer();
        buffer.order(ByteOrder.nativeOrder());
        for (int i = 0; i < data.length; i++) {
            int byteIndex = i / 8;
            int bitIndex = i % 8;
            byte byteValue = buffer.get(byteIndex);
            byteValue = (byte) (byteValue & ~(1 << bitIndex));
            if (data[i]) {
                byteValue = (byte) (byteValue | (1 << bitIndex));
            }
            buffer.put(byteIndex, byteValue);
        }
        buffer.flip();
        Pointer hostPtr = Pointer.to(buffer);
        cuCheck(cuMemcpyHtoD(deviceMemoryHandle.getDevicePointer(), hostPtr, size));
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

    @NotNull
    public CudaMemoryHandle getDeviceMemoryHandle() {
        return deviceMemoryHandle;
    }

    @Override
    public void fill(byte value) {
        cuCheck(cuMemsetD8(deviceMemoryHandle.getDevicePointer(), value, deviceMemoryHandle.getSize()));
    }

    @Override
    public void fill(short value) {
        if (deviceMemoryHandle.getSize() % Short.BYTES != 0) {
            throw new IllegalArgumentException("Cannot fill data container of size " + deviceMemoryHandle.getSize() + " with short value");
        }
        cuCheck(cuMemsetD16(deviceMemoryHandle.getDevicePointer(), value, deviceMemoryHandle.getSize() / Short.BYTES));
    }

    @Override
    public void fill(int value) {
        if (deviceMemoryHandle.getSize() % Integer.BYTES != 0) {
            throw new IllegalArgumentException("Cannot fill data container of size " + deviceMemoryHandle.getSize() + " with int value");
        }
        cuCheck(cuMemsetD32(deviceMemoryHandle.getDevicePointer(), value, deviceMemoryHandle.getSize() / Integer.BYTES));
    }

    @Override
    public void fill(long value) {
        if (deviceMemoryHandle.getSize() % Long.BYTES != 0) {
            throw new IllegalArgumentException("Cannot fill data container of size " + deviceMemoryHandle.getSize() + " with long value");
        }
        throw new UnsupportedOperationException("TODO: implement");
    }

    @Override
    public void fill(float value) {
        if (deviceMemoryHandle.getSize() % Float.BYTES != 0) {
            throw new IllegalArgumentException("Cannot fill data container of size " + deviceMemoryHandle.getSize() + " with float value");
        }
        cuCheck(cuMemsetD32(deviceMemoryHandle.getDevicePointer(), Float.floatToRawIntBits(value), deviceMemoryHandle.getSize() / Float.BYTES));
    }

    @Override
    public void fill(double value) {
        if (deviceMemoryHandle.getSize() % Double.BYTES != 0) {
            throw new IllegalArgumentException("Cannot fill data container of size " + deviceMemoryHandle.getSize() + " with double value");
        }
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
