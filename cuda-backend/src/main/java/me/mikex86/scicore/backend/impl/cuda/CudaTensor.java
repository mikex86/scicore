package me.mikex86.scicore.backend.impl.cuda;

import me.mikex86.scicore.AbstractTensor;
import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.ITensorIterator;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.jemalloc.JEmalloc;

import java.nio.*;
import java.util.Objects;

import static me.mikex86.scicore.backend.impl.cuda.Validator.cuCheck;
import static org.lwjgl.cuda.CU.*;

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

        private final long cuMemPtr;

        private final long size;

        public CudaDataContainer(long size) {
            this.size = size;
            try (MemoryStack stack = MemoryStack.stackPush()) {
                PointerBuffer dptrBuf = stack.mallocPointer(1);
                cuCheck(cuMemAlloc(dptrBuf, size));
                cuMemPtr = dptrBuf.get(0);
            }
        }

        public CudaDataContainer(long @NotNull [] shape, @NotNull DataType dataType) {
            this(getNumBytes(ShapeUtils.getNumElements(shape), dataType.getBits()));
        }

        private static long getNumBytes(long numElements, int bits) {
            return (numElements * bits + 7) / 8;
        }

        public byte getByteFlat(long flatIndex) {
            try (MemoryStack stack = MemoryStack.stackPush()) {
                ByteBuffer hostBuffer = stack.malloc(1);
                cuCheck(cuMemcpyDtoH(hostBuffer, cuMemPtr + flatIndex));
                return hostBuffer.get(0);
            }
        }

        public void setByteFlat(byte value, long flatIndex) {
            try (MemoryStack stack = MemoryStack.stackPush()) {
                ByteBuffer hostBuffer = stack.malloc(1);
                hostBuffer.put(0, value);
                cuCheck(cuMemcpyHtoD(cuMemPtr + flatIndex, hostBuffer));
            }
        }

        public short getShortFlat(long flatIndex) {
            try (MemoryStack stack = MemoryStack.stackPush()) {
                ShortBuffer hostBuffer = stack.mallocShort(1);
                cuCheck(cuMemcpyDtoH(hostBuffer, cuMemPtr + flatIndex * Short.BYTES));
                return hostBuffer.get(0);
            }
        }

        public void setShortFlat(short value, long flatIndex) {
            try (MemoryStack stack = MemoryStack.stackPush()) {
                ShortBuffer hostBuffer = stack.mallocShort(1);
                hostBuffer.put(0, value);
                cuCheck(cuMemcpyHtoD(cuMemPtr + flatIndex * Short.BYTES, hostBuffer));
            }
        }

        public int getIntFlat(long flatIndex) {
            try (MemoryStack stack = MemoryStack.stackPush()) {
                IntBuffer hostBuffer = stack.mallocInt(1);
                cuCheck(cuMemcpyDtoH(hostBuffer, cuMemPtr + flatIndex * Integer.BYTES));
                return hostBuffer.get(0);
            }
        }

        public void setIntFlat(int value, long flatIndex) {
            try (MemoryStack stack = MemoryStack.stackPush()) {
                IntBuffer hostBuffer = stack.mallocInt(1);
                hostBuffer.put(0, value);
                cuCheck(cuMemcpyHtoD(cuMemPtr + flatIndex * Integer.BYTES, hostBuffer));
            }
        }

        public long getLongFlat(long flatIndex) {
            try (MemoryStack stack = MemoryStack.stackPush()) {
                LongBuffer hostBuffer = stack.mallocLong(1);
                cuCheck(cuMemcpyDtoH(hostBuffer, cuMemPtr + flatIndex * Long.BYTES));
                return hostBuffer.get(0);
            }
        }

        public void setLongFlat(long value, long flatIndex) {
            try (MemoryStack stack = MemoryStack.stackPush()) {
                LongBuffer hostBuffer = stack.mallocLong(1);
                hostBuffer.put(0, value);
                cuCheck(cuMemcpyHtoD(cuMemPtr + flatIndex * Long.BYTES, hostBuffer));
            }
        }

        public float getFloatFlat(long flatIndex) {
            try (MemoryStack stack = MemoryStack.stackPush()) {
                FloatBuffer hostBuffer = stack.mallocFloat(1);
                cuCheck(cuMemcpyDtoH(hostBuffer, cuMemPtr + flatIndex * Float.BYTES));
                return hostBuffer.get(0);
            }
        }

        public void setFloatFlat(float value, long flatIndex) {
            try (MemoryStack stack = MemoryStack.stackPush()) {
                FloatBuffer hostBuffer = stack.mallocFloat(1);
                hostBuffer.put(0, value);
                cuCheck(cuMemcpyHtoD(cuMemPtr + flatIndex * Float.BYTES, hostBuffer));
            }
        }

        public double getDoubleFlat(long flatIndex) {
            try (MemoryStack stack = MemoryStack.stackPush()) {
                DoubleBuffer hostBuffer = stack.mallocDouble(1);
                cuCheck(cuMemcpyDtoH(hostBuffer, cuMemPtr + flatIndex * Double.BYTES));
                return hostBuffer.get(0);
            }
        }

        public void setDoubleFlat(double value, long flatIndex) {
            try (MemoryStack stack = MemoryStack.stackPush()) {
                DoubleBuffer hostBuffer = stack.mallocDouble(1);
                hostBuffer.put(0, value);
                cuCheck(cuMemcpyHtoD(cuMemPtr + flatIndex * Double.BYTES, hostBuffer));
            }
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
            if (!buffer.isDirect()) {
                // to direct buffer
                long size = buffer.remaining();
                ByteBuffer directBuffer = JEmalloc.je_malloc(size);
                if (directBuffer == null) {
                    throw new OutOfMemoryError("Could not allocate direct buffer of size " + size);
                }
                directBuffer.put(buffer);
                directBuffer.flip();
                buffer = directBuffer;
            }
            cuCheck(cuMemcpyHtoD(cuMemPtr, buffer));
            JEmalloc.je_free(buffer);
        }

        public void setContents(@NotNull ShortBuffer buffer) {
            if (buffer.remaining() > size / Short.BYTES) {
                throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
            }

            if (!buffer.isDirect()) {
                // to direct buffer
                long size = buffer.remaining() * (long) Short.BYTES;
                ShortBuffer directBuffer;
                {
                    ByteBuffer byteBuffer = JEmalloc.je_malloc(size);
                    if (byteBuffer == null) {
                        throw new OutOfMemoryError("Could not allocate direct buffer of size " + size);
                    }
                    directBuffer = byteBuffer.asShortBuffer();
                }
                directBuffer.put(buffer);
                directBuffer.flip();
                buffer = directBuffer;
            }
            cuCheck(cuMemcpyHtoD(cuMemPtr, buffer));
            JEmalloc.je_free(buffer);
        }

        public void setContents(@NotNull IntBuffer buffer) {
            if (buffer.remaining() > size / Integer.BYTES) {
                throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
            }
            if (!buffer.isDirect()) {
                // to direct buffer
                long size = buffer.remaining() * (long) Integer.BYTES;
                IntBuffer directBuffer;
                {
                    ByteBuffer byteBuffer = JEmalloc.je_malloc(size);
                    if (byteBuffer == null) {
                        throw new OutOfMemoryError("Could not allocate direct buffer of size " + size);
                    }
                    directBuffer = byteBuffer.asIntBuffer();
                }
                directBuffer.put(buffer);
                directBuffer.flip();
                buffer = directBuffer;
            }
            cuCheck(cuMemcpyHtoD(cuMemPtr, buffer));
            JEmalloc.je_free(buffer);
        }

        public void setContents(@NotNull LongBuffer buffer) {
            if (buffer.remaining() > size / Long.BYTES) {
                throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
            }
            if (!buffer.isDirect()) {
                // to direct buffer
                long size = buffer.remaining() * (long) Long.BYTES;
                LongBuffer directBuffer;
                {
                    ByteBuffer byteBuffer = JEmalloc.je_malloc(size);
                    if (byteBuffer == null) {
                        throw new OutOfMemoryError("Could not allocate direct buffer of size " + size);
                    }
                    directBuffer = byteBuffer.asLongBuffer();
                }
                directBuffer.put(buffer);
                directBuffer.flip();
                buffer = directBuffer;
            }
            cuCheck(cuMemcpyHtoD(cuMemPtr, buffer));
            JEmalloc.je_free(buffer);
        }

        public void setContents(@NotNull FloatBuffer buffer) {
            if (buffer.remaining() > size / Float.BYTES) {
                throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
            }
            if (!buffer.isDirect()) {
                // to direct buffer
                long size = buffer.remaining() * (long) Float.BYTES;
                FloatBuffer directBuffer;
                {
                    ByteBuffer byteBuffer = JEmalloc.je_malloc(size);
                    if (byteBuffer == null) {
                        throw new OutOfMemoryError("Could not allocate direct buffer of size " + size);
                    }
                    directBuffer = byteBuffer.asFloatBuffer();
                }
                directBuffer.put(buffer);
                directBuffer.flip();
                buffer = directBuffer;
            }
            cuCheck(cuMemcpyHtoD(cuMemPtr, buffer));
            JEmalloc.je_free(buffer);
        }

        public void setContents(@NotNull DoubleBuffer buffer) {
            if (buffer.remaining() > size / Double.BYTES) {
                throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
            }
            if (!buffer.isDirect()) {
                // to direct buffer
                long size = buffer.remaining() * (long) Double.BYTES;
                DoubleBuffer directBuffer;
                {
                    ByteBuffer byteBuffer = JEmalloc.je_malloc(size);
                    if (byteBuffer == null) {
                        throw new OutOfMemoryError("Could not allocate direct buffer of size " + size);
                    }
                    directBuffer = byteBuffer.asDoubleBuffer();
                }
                directBuffer.put(buffer);
                directBuffer.flip();
                buffer = directBuffer;
            }
            cuCheck(cuMemcpyHtoD(cuMemPtr, buffer));
            JEmalloc.je_free(buffer);
        }

        public void setContents(boolean @NotNull [] data) {
            if (data.length > size * 8) {
                throw new IllegalArgumentException("Cannot set contents of buffer, buffer is larger than data container size");
            }
            try (MemoryStack stack = MemoryStack.stackPush()) {
                PointerBuffer hostMemoryPtrBuf = stack.mallocPointer(1);
                long byteSize = (data.length + 7) / 8; // round up to next byte to have enough space for n bits
                cuCheck(cuMemAllocHost(hostMemoryPtrBuf, byteSize));
                ByteBuffer byteBuffer = hostMemoryPtrBuf.getByteBuffer(0, Math.toIntExact(byteSize));
                for (int i = 0; i < data.length; i++) {
                    long byteIndex = i / 8;
                    int bitIndex = i % 8;
                    byte byteValue = byteBuffer.get((int) byteIndex);
                    byteValue = (byte) (byteValue & ~(1 << bitIndex));
                    if (data[i]) {
                        byteValue = (byte) (byteValue | (1 << bitIndex));
                    }
                    byteBuffer.put((int) byteIndex, byteValue);
                }
                cuCheck(cuMemcpyHtoD(cuMemPtr, byteBuffer));
            }
        }

        @NotNull
        public ByteBuffer getContents() {
            if (size > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Contents larger than 2^31-1 bytes cannot be copied to the host");
            }
            try (MemoryStack stack = MemoryStack.stackPush()) {
                PointerBuffer hostMemoryPtrBuf = stack.mallocPointer(1);
                cuCheck(cuMemAllocHost(hostMemoryPtrBuf, size));
                cuCheck(cuMemcpyDtoH(hostMemoryPtrBuf, cuMemPtr));
                return hostMemoryPtrBuf.getByteBuffer(0, Math.toIntExact(size));
            }
        }
    }
}
