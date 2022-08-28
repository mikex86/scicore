package me.mikex86.scicore.backend.impl.cuda;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import me.mikex86.scicore.AbstractTensor;
import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.ITensorIterator;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.nio.*;

import static jcuda.driver.JCudaDriver.*;
import static me.mikex86.scicore.backend.impl.cuda.Validator.cuCheck;

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
        this.backend = backend;
        this.dataType = dataType;
        this.shape = shape;
        this.strides = ShapeUtils.makeStrides(shape);
        this.dataContainer = new CudaDataContainer(shape, dataType);
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
        return dataContainer.getByteFlat(flatIndex);
    }

    @Override
    public void setByteFlat(byte value, long flatIndex) {
        dataContainer.setByteFlat(value, flatIndex);
    }

    @Override
    public short getShortFlat(long flatIndex) {
        return dataContainer.getShortFlat(flatIndex);
    }

    @Override
    public void setShortFlat(short value, long flatIndex) {
        dataContainer.setShortFlat(value, flatIndex);
    }

    @Override
    public int getIntFlat(long flatIndex) {
        return dataContainer.getIntFlat(flatIndex);
    }

    @Override
    public void setIntFlat(int value, long flatIndex) {
        dataContainer.setIntFlat(value, flatIndex);
    }

    @Override
    public long getLongFlat(long flatIndex) {
        return dataContainer.getLongFlat(flatIndex);
    }

    @Override
    public void setLongFlat(long value, long flatIndex) {
        dataContainer.setLongFlat(value, flatIndex);
    }


    @Override
    public float getFloatFlat(long flatIndex) {
        return dataContainer.getFloatFlat(flatIndex);
    }

    @Override
    public void setFloatFlat(float value, long flatIndex) {
        dataContainer.setFloatFlat(value, flatIndex);
    }

    @Override
    public double getDoubleFlat(long flatIndex) {
        return dataContainer.getDoubleFlat(flatIndex);
    }

    @Override
    public void setDoubleFlat(double value, long flatIndex) {
        dataContainer.setDoubleFlat(value, flatIndex);
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
        return null;
    }

    @Override
    public void setContents(@NotNull ITensor tensor) {
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

    }

    @Override
    public void fill(byte value) {

    }

    @Override
    public void fill(short value) {

    }

    @Override
    public void fill(int value) {

    }

    @Override
    public void fill(long value) {

    }

    @Override
    public void fill(float value) {

    }

    @Override
    public void fill(double value) {

    }

    @Override
    public void fill(boolean value) {

    }

    @Override
    public @NotNull ITensorIterator iterator() {
        return null;
    }

    @Override
    public @NotNull ISciCoreBackend getSciCoreBackend() {
        return backend;
    }

    private static class CudaDataContainer {

        @NotNull
        private final CUdeviceptr cuMemPtr;

        private final long size;

        public CudaDataContainer(long size) {
            this.size = size;

            CUdeviceptr dptrBuf = new CUdeviceptr();
            cuCheck(cuMemAlloc(dptrBuf, size));
            cuMemPtr = dptrBuf;
        }

        public CudaDataContainer(long @NotNull [] shape, @NotNull DataType dataType) {
            this(getNumBytes(ShapeUtils.getNumElements(shape), dataType.getBits()));
        }

        private static long getNumBytes(long numElements, int bits) {
            return (numElements * bits + 7) / 8;
        }

        public byte getByteFlat(long flatIndex) {
            byte[] hostBuffer = new byte[1];
            Pointer hostPtr = Pointer.to(hostBuffer);
            cuCheck(cuMemcpyDtoH(hostPtr, cuMemPtr.withByteOffset(flatIndex), 1));
            return hostBuffer[0];
        }

        public void setByteFlat(byte value, long flatIndex) {
            byte[] hostBuffer = new byte[]{value};
            Pointer hostPtr = Pointer.to(hostBuffer);
            cuCheck(cuMemcpyHtoD(cuMemPtr.withByteOffset(flatIndex), hostPtr, 1));
        }

        public short getShortFlat(long flatIndex) {
            short[] hostBuffer = new short[1];
            Pointer hostPtr = Pointer.to(hostBuffer);
            cuCheck(cuMemcpyDtoH(hostPtr, cuMemPtr.withByteOffset(flatIndex * Short.BYTES), Short.BYTES));
            return hostBuffer[0];
        }

        public void setShortFlat(short value, long flatIndex) {
            short[] hostBuffer = new short[]{value};
            Pointer hostPtr = Pointer.to(hostBuffer);
            cuCheck(cuMemcpyHtoD(cuMemPtr.withByteOffset(flatIndex * Short.BYTES), hostPtr, Short.BYTES));
        }

        public int getIntFlat(long flatIndex) {
            int[] hostBuffer = new int[1];
            Pointer hostPtr = Pointer.to(hostBuffer);
            cuCheck(cuMemcpyDtoH(hostPtr, cuMemPtr.withByteOffset(flatIndex * Integer.BYTES), Integer.BYTES));
            return hostBuffer[0];
        }

        public void setIntFlat(int value, long flatIndex) {
            int[] hostBuffer = new int[]{value};
            Pointer hostPtr = Pointer.to(hostBuffer);
            cuCheck(cuMemcpyHtoD(cuMemPtr.withByteOffset(flatIndex * Integer.BYTES), hostPtr, Integer.BYTES));
        }

        public long getLongFlat(long flatIndex) {
            long[] hostBuffer = new long[1];
            Pointer hostPtr = Pointer.to(hostBuffer);
            cuCheck(cuMemcpyDtoH(hostPtr, cuMemPtr.withByteOffset(flatIndex * Long.BYTES), Long.BYTES));
            return hostBuffer[0];
        }

        public void setLongFlat(long value, long flatIndex) {
            long[] hostBuffer = new long[]{value};
            Pointer hostPtr = Pointer.to(hostBuffer);
            cuCheck(cuMemcpyHtoD(cuMemPtr.withByteOffset(flatIndex * Long.BYTES), hostPtr, Long.BYTES));
        }

        public float getFloatFlat(long flatIndex) {
            float[] hostBuffer = new float[1];
            Pointer hostPtr = Pointer.to(hostBuffer);
            cuCheck(cuMemcpyDtoH(hostPtr, cuMemPtr.withByteOffset(flatIndex * Float.BYTES), Float.BYTES));
            return hostBuffer[0];
        }

        public void setFloatFlat(float value, long flatIndex) {
            float[] hostBuffer = new float[]{value};
            Pointer hostPtr = Pointer.to(hostBuffer);
            cuCheck(cuMemcpyHtoD(cuMemPtr.withByteOffset(flatIndex * Float.BYTES), hostPtr, Float.BYTES));
        }

        public double getDoubleFlat(long flatIndex) {
            double[] hostBuffer = new double[1];
            Pointer hostPtr = Pointer.to(hostBuffer);
            cuCheck(cuMemcpyDtoH(hostPtr, cuMemPtr.withByteOffset(flatIndex * Double.BYTES), Double.BYTES));
            return hostBuffer[0];
        }

        public void setDoubleFlat(double value, long flatIndex) {
            double[] hostBuffer = new double[]{value};
            Pointer hostPtr = Pointer.to(hostBuffer);
            cuCheck(cuMemcpyHtoD(cuMemPtr.withByteOffset(flatIndex * Double.BYTES), hostPtr, Double.BYTES));
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
            if (buffer.remaining() > size) {
                throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
            }
            int size = buffer.remaining();
            if (!buffer.isDirect()) {
                // to direct buffer
                ByteBuffer directBuffer = ByteBuffer
                        .allocateDirect(size)
                        .order(ByteOrder.nativeOrder());
                directBuffer.put(buffer);
                directBuffer.flip();
                buffer = directBuffer;
            }
            Pointer hostPtr = Pointer.to(buffer);
            cuCheck(cuMemcpyHtoD(cuMemPtr, hostPtr, size));
        }

        public void setContents(@NotNull ShortBuffer buffer) {
            if (buffer.remaining() > size / Short.BYTES) {
                throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
            }

            int size = buffer.remaining() * Short.BYTES;
            if (!buffer.isDirect()) {
                // to direct buffer
                ShortBuffer directBuffer = ByteBuffer
                        .allocateDirect(size)
                        .order(ByteOrder.nativeOrder())
                        .asShortBuffer();
                directBuffer.put(buffer);
                directBuffer.flip();
                buffer = directBuffer;
            }
            Pointer hostPtr = Pointer.to(buffer);
            cuCheck(cuMemcpyHtoD(cuMemPtr, hostPtr, size));
        }

        public void setContents(@NotNull IntBuffer buffer) {
            if (buffer.remaining() > size / Integer.BYTES) {
                throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
            }

            int size = buffer.remaining() * Integer.BYTES;
            if (!buffer.isDirect()) {
                // to direct buffer
                IntBuffer directBuffer = ByteBuffer
                        .allocateDirect(size)
                        .order(ByteOrder.nativeOrder())
                        .asIntBuffer();
                directBuffer.put(buffer);
                directBuffer.flip();
                buffer = directBuffer;
            }
            Pointer hostPtr = Pointer.to(buffer);
            cuCheck(cuMemcpyHtoD(cuMemPtr, hostPtr, size));
        }

        public void setContents(@NotNull LongBuffer buffer) {
            if (buffer.remaining() > size / Long.BYTES) {
                throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
            }

            int size = buffer.remaining() * Long.BYTES;
            if (!buffer.isDirect()) {
                // to direct buffer
                LongBuffer directBuffer = ByteBuffer
                        .allocateDirect(size)
                        .order(ByteOrder.nativeOrder())
                        .asLongBuffer();
                directBuffer.put(buffer);
                directBuffer.flip();
                buffer = directBuffer;
            }
            Pointer hostPtr = Pointer.to(buffer);
            cuCheck(cuMemcpyHtoD(cuMemPtr, hostPtr, size));
        }

        public void setContents(@NotNull FloatBuffer buffer) {
            if (buffer.remaining() > size / Float.BYTES) {
                throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
            }

            int size = buffer.remaining() * Float.BYTES;
            if (!buffer.isDirect()) {
                // to direct buffer
                FloatBuffer directBuffer = ByteBuffer
                        .allocateDirect(size)
                        .order(ByteOrder.nativeOrder())
                        .asFloatBuffer();
                directBuffer.put(buffer);
                directBuffer.flip();
                buffer = directBuffer;
            }
            Pointer hostPtr = Pointer.to(buffer);
            cuCheck(cuMemcpyHtoD(cuMemPtr, hostPtr, size));
        }

        public void setContents(@NotNull DoubleBuffer buffer) {
            if (buffer.remaining() > size / Double.BYTES) {
                throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
            }

            int size = buffer.remaining() * Double.BYTES;
            if (!buffer.isDirect()) {
                // to direct buffer
                DoubleBuffer directBuffer = ByteBuffer
                        .allocateDirect(size)
                        .order(ByteOrder.nativeOrder())
                        .asDoubleBuffer();
                directBuffer.put(buffer);
                directBuffer.flip();
                buffer = directBuffer;
            }
            Pointer hostPtr = Pointer.to(buffer);
            cuCheck(cuMemcpyHtoD(cuMemPtr, hostPtr, size));
        }

        public void setContents(boolean @NotNull [] data) {
            if (data.length > size * 8) {
                throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
            }
            int size = (data.length + 7) / 8; // round up to next byte
            ByteBuffer buffer = ByteBuffer
                    .allocateDirect(size)
                    .order(ByteOrder.nativeOrder());
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
            cuCheck(cuMemcpyHtoD(cuMemPtr, hostPtr, size));
        }

        @NotNull
        public ByteBuffer getContents() {
            if (size > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Contents larger than 2^31-1 bytes cannot be copied to the host");
            }
            ByteBuffer buffer = ByteBuffer.allocateDirect((int) size);
            Pointer hostPtr = Pointer.to(buffer);
            cuCheck(cuMemcpyDtoH(hostPtr, cuMemPtr, size));
            return buffer;
        }
    }
}
