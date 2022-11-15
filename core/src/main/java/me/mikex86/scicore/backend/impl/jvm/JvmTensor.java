package me.mikex86.scicore.backend.impl.jvm;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.tensor.AbstractTensor;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.data.ITensorDataContainer;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import me.mikex86.scicore.utils.dispose.IDisposable;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.*;
import java.nio.*;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Objects;

public class JvmTensor extends AbstractTensor implements ITensor {

    @NotNull
    private final JvmTensorDataContainer dataContainer;

    private final long @NotNull [] strides;

    @NotNull
    private final JvmBackend backend;


    public JvmTensor(@NotNull JvmBackend backend, @NotNull DataType dataType, long @NotNull [] shape) {
        this.numElements = ShapeUtils.getNumElements(shape);
        this.dataContainer = new JvmTensorDataContainer(backend, dataType, shape);
        this.strides = ShapeUtils.makeStrides(shape);
        this.backend = backend;
    }

    JvmTensor(@NotNull JvmBackend backend, @NotNull JvmTensorDataContainer dataContainer, long @NotNull [] shape) {
        this.numElements = ShapeUtils.getNumElements(shape);
        this.backend = backend;
        this.dataContainer = dataContainer;
        this.strides = ShapeUtils.makeStrides(shape);
    }

    @Override
    public @NotNull
    DataType getDataType() {
        return this.dataContainer.getDataType();
    }

    @Override
    public long @NotNull [] getShape() {
        return this.dataContainer.getShape();
    }

    @Override
    public long @NotNull [] getStrides() {
        return this.strides;
    }

    @Override
    public boolean getBooleanFlat(long flatIndex) {
        return this.dataContainer.getBooleanFlat(flatIndex);
    }

    @Override
    public void setBooleanFlat(boolean value, long flatIndex) {
        this.dataContainer.setBooleanFlat(flatIndex, value);
    }

    @Override
    public byte getByteFlat(long flatIndex) {
        return this.dataContainer.getInt8Flat(flatIndex);
    }

    @Override
    public void setByteFlat(byte value, long flatIndex) {
        this.dataContainer.setInt8Flat(flatIndex, value);
    }

    @Override
    public short getShortFlat(long flatIndex) {
        return this.dataContainer.getInt16Flat(flatIndex);
    }

    @Override
    public void setShortFlat(short value, long flatIndex) {
        this.dataContainer.setInt16Flat(flatIndex, value);
    }

    @Override
    public int getIntFlat(long flatIndex) {
        return this.dataContainer.getInt32Flat(flatIndex);
    }

    @Override
    public void setIntFlat(int value, long flatIndex) {
        this.dataContainer.setInt32Flat(flatIndex, value);
    }

    @Override
    public long getLongFlat(long flatIndex) {
        return this.dataContainer.getInt64Flat(flatIndex);
    }

    @Override
    public void setLongFlat(long value, long flatIndex) {
        this.dataContainer.setInt64Flat(flatIndex, value);
    }

    @Override
    public float getFloatFlat(long flatIndex) {
        return this.dataContainer.getFloat32Flat(flatIndex);
    }

    @Override
    public void setFloatFlat(float value, long flatIndex) {
        this.dataContainer.setFloat32Flat(flatIndex, value);
    }

    @Override
    public double getDoubleFlat(long flatIndex) {
        return this.dataContainer.getFloat64Flat(flatIndex);
    }

    @Override
    public void setDoubleFlat(double value, long flatIndex) {
        this.dataContainer.setFloat64Flat(flatIndex, value);
    }

    @Override
    public @NotNull ITensor copy() {
        ITensor copy = this.backend.createTensor(this.dataContainer.getDataType(), this.dataContainer.getShape());
        copy.setContents(this);
        return copy;
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull ITensor tensor) {
        DataType dataType = getDataType();
        long numElements = tensor.getNumberOfElements();
        if (numElements > this.numElements - startFlatIndex) {
            throw new IllegalArgumentException("Tensor is too large to fit in this tensor");
        }
        Validator.assertTrue(dataType == tensor.getDataType(), "Cannot set contents of tensor with data type " + dataType + " to tensor with data type " + tensor.getDataType());
        if (tensor instanceof JvmTensor jvmTensor) {
            this.dataContainer.setContents(Math.toIntExact(startFlatIndex), jvmTensor.dataContainer);
        } else { // General copy
            long nElementsToCopy = tensor.getNumberOfElements();
            for (long i = 0; i < nElementsToCopy; i++) {
                switch (this.getDataType()) {
                    case INT8 -> this.setByteFlat(tensor.getByteFlat(i), startFlatIndex + i);
                    case INT16 -> this.setShortFlat(tensor.getShortFlat(i), startFlatIndex + i);
                    case INT32 -> this.setIntFlat(tensor.getIntFlat(i), startFlatIndex + i);
                    case INT64 -> this.setLongFlat(tensor.getLongFlat(i), startFlatIndex + i);
                    case FLOAT32 -> this.setFloatFlat(tensor.getFloatFlat(i), startFlatIndex + i);
                    case FLOAT64 -> this.setDoubleFlat(tensor.getDoubleFlat(i), startFlatIndex + i);
                    default -> throw new IllegalArgumentException("Unsupported data type");
                }
            }
        }
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull ByteBuffer buffer) {
        // set with bytes can be called no matter the data type, while for other types this will fail.
        // so here we have to convert element-level indices to byte-level indices.
        long byteIndex = getDataType().getSizeOf(startFlatIndex);
        this.dataContainer.setContents(byteIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull ShortBuffer buffer) {
        validateDataType(DataType.INT16);
        this.dataContainer.setContents(startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull IntBuffer buffer) {
        validateDataType(DataType.INT32);
        this.dataContainer.setContents(startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull LongBuffer buffer) {
        validateDataType(DataType.INT64);
        this.dataContainer.setContents(startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull FloatBuffer buffer) {
        validateDataType(DataType.FLOAT32);
        this.dataContainer.setContents(startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull DoubleBuffer buffer) {
        validateDataType(DataType.FLOAT64);
        this.dataContainer.setContents(startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, boolean @NotNull [] buffer) {
        validateDataType(DataType.BOOLEAN);
        this.dataContainer.setContents(startFlatIndex, buffer);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, byte value) {
        this.dataContainer.fillRegion(startFlatIndex, endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, short value) {
        this.dataContainer.fillRegion(startFlatIndex, endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, int value) {
        this.dataContainer.fillRegion(startFlatIndex, endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, long value) {
        this.dataContainer.fillRegion(startFlatIndex, endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, float value) {
        this.dataContainer.fillRegion(startFlatIndex, endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, double value) {
        this.dataContainer.fillRegion(startFlatIndex, endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, boolean value) {
        this.dataContainer.fillRegion(startFlatIndex, endFlatIndex, value);
    }

    @Override
    public void writeTo(@NotNull OutputStream outputStream) throws IOException {
        DataOutputStream dataOut = new DataOutputStream(outputStream);
        switch (getDataType()) {
            case INT8 -> dataOut.write(dataContainer.getByteData());
            case INT16 -> {
                short[] shortData = dataContainer.getShortData();
                for (short s : shortData) {
                    dataOut.writeShort(s);
                }
            }
            case INT32 -> {
                int[] intData = dataContainer.getIntData();
                for (int i : intData) {
                    dataOut.writeInt(i);
                }
            }
            case INT64 -> {
                long[] longData = dataContainer.getLongData();
                for (long l : longData) {
                    dataOut.writeLong(l);
                }
            }
            case FLOAT32 -> {
                float[] floatData = dataContainer.getFloatData();
                for (float f : floatData) {
                    dataOut.writeFloat(f);
                }
            }
            case FLOAT64 -> {
                double[] doubleData = dataContainer.getDoubleData();
                for (double d : doubleData) {
                    dataOut.writeDouble(d);
                }
            }
            case BOOLEAN -> {
                BitSet booleanData = dataContainer.getBooleanData();
                byte[] bytes = booleanData.toByteArray();
                dataOut.write(bytes);
            }
        }
    }


    private static class JvmTensorDataContainer implements ITensorDataContainer, IDisposable {

        @NotNull
        private final JvmBackend backend;

        @NotNull
        private final DataType dataType;

        private final long @NotNull [] shape;

        private final int nElements;

        private byte @Nullable [] byteData;
        private short @Nullable [] shortData;
        private int @Nullable [] intData;
        private long @Nullable [] longData;
        private float @Nullable [] floatData;
        private double @Nullable [] doubleData;

        private @Nullable BitSet bitSetData;
        private int nBits;

        private JvmTensorDataContainer(@NotNull JvmBackend backend, @NotNull DataType dataType, long @NotNull [] shape) {
            this.backend = backend;
            this.dataType = dataType;
            this.shape = shape;
            this.nElements = Math.toIntExact(ShapeUtils.getNumElements(shape));
            switch (dataType) {
                case INT8 -> this.byteData = new byte[nElements];
                case INT16 -> this.shortData = new short[nElements];
                case INT32 -> this.intData = new int[nElements];
                case INT64 -> this.longData = new long[nElements];
                case FLOAT32 -> this.floatData = new float[nElements];
                case FLOAT64 -> this.doubleData = new double[nElements];
                case BOOLEAN -> {
                    this.nBits = nElements;
                    this.bitSetData = new BitSet(nBits);
                }
            }
        }

        byte @NotNull [] getByteData() {
            return Objects.requireNonNull(byteData, "DataContainer has different data type");
        }

        short @NotNull [] getShortData() {
            return Objects.requireNonNull(shortData, "DataContainer has different data type");
        }

        int @NotNull [] getIntData() {
            return Objects.requireNonNull(intData, "DataContainer has different data type");
        }

        float @NotNull [] getFloatData() {
            return Objects.requireNonNull(floatData, "DataContainer has different data type");
        }

        double @NotNull [] getDoubleData() {
            return Objects.requireNonNull(doubleData, "DataContainer has different data type");
        }

        long @NotNull [] getLongData() {
            return Objects.requireNonNull(longData, "DataContainer has different data type");
        }

        @NotNull BitSet getBooleanData() {
            return Objects.requireNonNull(bitSetData, "DataContainer has different data type");
        }

        @Override
        public byte getInt8Flat(long index) {
            byte[] byteData = getByteData();
            if (index < 0 || index >= byteData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            return byteData[(int) index];
        }

        @Override
        public short getInt16Flat(long index) {
            short[] shortData = getShortData();
            if (index < 0 || index >= shortData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            return shortData[(int) index];
        }

        @Override
        public int getInt32Flat(long index) {
            int[] intData = getIntData();
            if (index < 0 || index >= intData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            return intData[(int) index];
        }

        @Override
        public long getInt64Flat(long index) {
            long[] longData = getLongData();
            if (index < 0 || index >= longData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            return longData[(int) index];
        }

        @Override
        public float getFloat32Flat(long index) {
            float[] floatData = getFloatData();
            if (index < 0 || index >= floatData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            return floatData[(int) index];
        }

        @Override
        public double getFloat64Flat(long index) {
            double[] doubleData = getDoubleData();
            if (index < 0 || index >= doubleData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            return doubleData[(int) index];
        }

        @Override
        public boolean getBooleanFlat(long index) {
            BitSet bitSetData = getBooleanData();
            if (index < 0 || index >= nBits) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            return bitSetData.get((int) index);
        }

        @Override
        public void setInt8Flat(long index, byte value) {
            byte[] byteData = getByteData();
            if (index < 0 || index >= byteData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            byteData[(int) index] = value;
        }

        @Override
        public void setInt16Flat(long index, short value) {
            short[] shortData = getShortData();
            if (index < 0 || index >= shortData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            shortData[(int) index] = value;
        }

        @Override
        public void setInt32Flat(long index, int value) {
            int[] intData = getIntData();
            if (index < 0 || index >= intData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            intData[(int) index] = value;
        }

        @Override
        public void setInt64Flat(long index, long value) {
            long[] longData = getLongData();
            if (index < 0 || index >= longData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            longData[(int) index] = value;
        }

        @Override
        public void setFloat32Flat(long index, float value) {
            float[] floatData = getFloatData();
            if (index < 0 || index >= floatData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            floatData[(int) index] = value;
        }

        @Override
        public void setFloat64Flat(long index, double value) {
            double[] doubleData = getDoubleData();
            if (index < 0 || index >= doubleData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            doubleData[(int) index] = value;
        }

        @Override
        public void setBooleanFlat(long index, boolean value) {
            BitSet booleanData = getBooleanData();
            if (index < 0 || index >= nBits) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            booleanData.set(Math.toIntExact(index), value);
        }

        @NotNull
        public DataType getDataType() {
            return dataType;
        }

        @Override
        public long getDataSize() {
            return dataType.getSizeOf(nElements);
        }

        @Override
        public long getNumberOfElements() {
            return 0;
        }

        public long @NotNull [] getShape() {
            return shape;
        }

        @Override
        public void setContents(long startIndex, @NotNull ByteBuffer buffer) {
            if (dataType == DataType.BOOLEAN) {
                throw new UnsupportedOperationException("Cannot set contents of boolean data");
            }
            if (buffer.remaining() % dataType.getSize() != 0) {
                throw new IllegalArgumentException("Buffer size is not a multiple of data type size");
            }
            if (startIndex < 0 || startIndex >= nElements) {
                throw new IndexOutOfBoundsException("Start index " + startIndex + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            if (startIndex + buffer.remaining() / dataType.getSize() > nElements) {
                throw new IndexOutOfBoundsException("Buffer size is too large for shape " + ShapeUtils.toString(shape));
            }
            switch (dataType) {
                case INT8 ->
                        buffer.asReadOnlyBuffer().get(getByteData(), Math.toIntExact(startIndex), buffer.remaining());
                case INT16 ->
                        buffer.asShortBuffer().get(getShortData(), Math.toIntExact(startIndex), buffer.remaining() / dataType.getSize());
                case INT32 ->
                        buffer.asIntBuffer().get(getIntData(), Math.toIntExact(startIndex), buffer.remaining() / dataType.getSize());
                case INT64 ->
                        buffer.asLongBuffer().get(getLongData(), Math.toIntExact(startIndex), buffer.remaining() / dataType.getSize());
                case FLOAT32 ->
                        buffer.asFloatBuffer().get(getFloatData(), Math.toIntExact(startIndex), buffer.remaining() / dataType.getSize());
                case FLOAT64 ->
                        buffer.asDoubleBuffer().get(getDoubleData(), Math.toIntExact(startIndex), buffer.remaining() / dataType.getSize());
                default -> throw new UnsupportedOperationException("Unsupported data type " + dataType);
            }
        }

        @Override
        public void setContents(long startIndex, @NotNull ShortBuffer buffer) {
            if (dataType != DataType.INT16) {
                throw new UnsupportedOperationException("Cannot set contents of a DataContainer with data type " + dataType);
            }
            short[] shortData = getShortData();
            if (startIndex < 0 || startIndex >= shortData.length) {
                throw new IndexOutOfBoundsException("Start index " + startIndex + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            if (startIndex + buffer.remaining() > shortData.length) {
                throw new IndexOutOfBoundsException("Buffer size is too large for shape " + ShapeUtils.toString(shape));
            }
            buffer.get(shortData, Math.toIntExact(startIndex), buffer.remaining());
        }

        @Override
        public void setContents(long startIndex, @NotNull IntBuffer buffer) {
            if (dataType != DataType.INT32) {
                throw new UnsupportedOperationException("Cannot set contents of a DataContainer with data type " + dataType);
            }
            int[] intData = getIntData();
            if (startIndex < 0 || startIndex >= intData.length) {
                throw new IndexOutOfBoundsException("Start index " + startIndex + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            if (startIndex + buffer.remaining() > intData.length) {
                throw new IndexOutOfBoundsException("Buffer size is too large for shape " + ShapeUtils.toString(shape));
            }
            buffer.get(intData, Math.toIntExact(startIndex), buffer.remaining());
        }

        @Override
        public void setContents(long startIndex, @NotNull LongBuffer buffer) {
            if (dataType != DataType.INT64) {
                throw new UnsupportedOperationException("Cannot set contents of a DataContainer with data type " + dataType);
            }
            long[] longData = getLongData();
            if (startIndex < 0 || startIndex >= longData.length) {
                throw new IndexOutOfBoundsException("Start index " + startIndex + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            if (startIndex + buffer.remaining() > longData.length) {
                throw new IndexOutOfBoundsException("Buffer size is too large for shape " + ShapeUtils.toString(shape));
            }
            buffer.get(longData, Math.toIntExact(startIndex), buffer.remaining());
        }

        @Override
        public void setContents(long startIndex, @NotNull FloatBuffer buffer) {
            if (dataType != DataType.FLOAT32) {
                throw new UnsupportedOperationException("Cannot set contents of a DataContainer with data type " + dataType);
            }
            float[] floatData = getFloatData();
            if (startIndex < 0 || startIndex >= floatData.length) {
                throw new IndexOutOfBoundsException("Start index " + startIndex + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            if (startIndex + buffer.remaining() > floatData.length) {
                throw new IndexOutOfBoundsException("Buffer size is too large for shape " + ShapeUtils.toString(shape));
            }
            buffer.get(floatData, Math.toIntExact(startIndex), buffer.remaining());
        }

        @Override
        public void setContents(long startIndex, @NotNull DoubleBuffer buffer) {
            if (dataType != DataType.FLOAT64) {
                throw new UnsupportedOperationException("Cannot set contents of a DataContainer with data type " + dataType);
            }
            double[] doubleData = getDoubleData();
            if (startIndex < 0 || startIndex >= doubleData.length) {
                throw new IndexOutOfBoundsException("Start index " + startIndex + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            if (startIndex + buffer.remaining() > doubleData.length) {
                throw new IndexOutOfBoundsException("Buffer size is too large for shape " + ShapeUtils.toString(shape));
            }
            buffer.get(doubleData, Math.toIntExact(startIndex), buffer.remaining());
        }

        @Override
        public void setContents(long startIndex, boolean @NotNull [] data) {
            if (dataType != DataType.BOOLEAN) {
                throw new UnsupportedOperationException("Cannot set contents of a DataContainer with data type " + dataType);
            }
            BitSet booleanData = getBooleanData();
            if (startIndex < 0 || startIndex >= nElements) {
                throw new IndexOutOfBoundsException("Start index " + startIndex + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            if (startIndex + data.length > nElements) {
                throw new IndexOutOfBoundsException("Buffer size is too large for shape " + ShapeUtils.toString(shape));
            }
            for (int i = 0; i < data.length; i++) {
                booleanData.set(Math.toIntExact(startIndex + i), data[i]);
            }
        }

        @Override
        public void fillRegion(long startFlatIndex, long endFlatIndex, byte value) {
            if (startFlatIndex < 0 || startFlatIndex >= nElements) {
                throw new IndexOutOfBoundsException("Start index " + startFlatIndex + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            if (endFlatIndex < 0 || endFlatIndex > nElements) {
                throw new IndexOutOfBoundsException("End index " + endFlatIndex + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            switch (dataType) {
                case INT8 -> {
                    byte[] byteData = getByteData();
                    Arrays.fill(byteData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value);
                }
                case INT16 -> {
                    short[] shortData = getShortData();
                    Arrays.fill(shortData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value);
                }
                case INT32 -> {
                    int[] intData = getIntData();
                    Arrays.fill(intData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value);
                }
                case INT64 -> {
                    long[] longData = getLongData();
                    Arrays.fill(longData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value);
                }
                case FLOAT32 -> {
                    float[] floatData = getFloatData();
                    Arrays.fill(floatData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value);
                }
                case FLOAT64 -> {
                    double[] doubleData = getDoubleData();
                    Arrays.fill(doubleData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value);
                }
                case BOOLEAN -> {
                    BitSet booleanData = getBooleanData();
                    for (long i = startFlatIndex; i < endFlatIndex; i++) {
                        booleanData.set(Math.toIntExact(i), value != 0);
                    }
                }
            }
        }

        @Override
        public void fillRegion(long startFlatIndex, long endFlatIndex, short value) {
            if (startFlatIndex < 0 || startFlatIndex >= nElements) {
                throw new IndexOutOfBoundsException("Start index " + startFlatIndex + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            if (endFlatIndex < 0 || endFlatIndex > nElements) {
                throw new IndexOutOfBoundsException("End index " + endFlatIndex + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            switch (dataType) {
                case INT8 -> {
                    byte[] byteData = getByteData();
                    Arrays.fill(byteData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), (byte) value);
                }
                case INT16 -> {
                    short[] shortData = getShortData();
                    Arrays.fill(shortData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value);
                }
                case INT32 -> {
                    int[] intData = getIntData();
                    Arrays.fill(intData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value);
                }
                case INT64 -> {
                    long[] longData = getLongData();
                    Arrays.fill(longData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value);
                }
                case FLOAT32 -> {
                    float[] floatData = getFloatData();
                    Arrays.fill(floatData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value);
                }
                case FLOAT64 -> {
                    double[] doubleData = getDoubleData();
                    Arrays.fill(doubleData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value);
                }
                case BOOLEAN -> {
                    BitSet booleanData = getBooleanData();
                    for (long i = startFlatIndex; i < endFlatIndex; i++) {
                        booleanData.set(Math.toIntExact(i), value != 0);
                    }
                }
            }
        }

        @Override
        public void fillRegion(long startFlatIndex, long endFlatIndex, int value) {
            if (startFlatIndex < 0 || startFlatIndex >= nElements) {
                throw new IndexOutOfBoundsException("Start index " + startFlatIndex + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            if (endFlatIndex < 0 || endFlatIndex > nElements) {
                throw new IndexOutOfBoundsException("End index " + endFlatIndex + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            switch (dataType) {
                case INT8 -> {
                    byte[] byteData = getByteData();
                    Arrays.fill(byteData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), (byte) value);
                }
                case INT16 -> {
                    short[] shortData = getShortData();
                    Arrays.fill(shortData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), (short) value);
                }
                case INT32 -> {
                    int[] intData = getIntData();
                    Arrays.fill(intData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value);
                }
                case INT64 -> {
                    long[] longData = getLongData();
                    Arrays.fill(longData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value);
                }
                case FLOAT32 -> {
                    float[] floatData = getFloatData();
                    Arrays.fill(floatData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value);
                }
                case FLOAT64 -> {
                    double[] doubleData = getDoubleData();
                    Arrays.fill(doubleData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value);
                }
                case BOOLEAN -> {
                    BitSet booleanData = getBooleanData();
                    for (long i = startFlatIndex; i < endFlatIndex; i++) {
                        booleanData.set(Math.toIntExact(i), value != 0);
                    }
                }
            }
        }

        @Override
        public void fillRegion(long startFlatIndex, long endFlatIndex, long value) {
            if (startFlatIndex < 0 || startFlatIndex >= nElements) {
                throw new IndexOutOfBoundsException("Start index " + startFlatIndex + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            if (endFlatIndex < 0 || endFlatIndex > nElements) {
                throw new IndexOutOfBoundsException("End index " + endFlatIndex + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            switch (dataType) {
                case INT8 -> {
                    byte[] byteData = getByteData();
                    Arrays.fill(byteData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), (byte) value);
                }
                case INT16 -> {
                    short[] shortData = getShortData();
                    Arrays.fill(shortData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), (short) value);
                }
                case INT32 -> {
                    int[] intData = getIntData();
                    Arrays.fill(intData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), (int) value);
                }
                case INT64 -> {
                    long[] longData = getLongData();
                    Arrays.fill(longData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value);
                }
                case FLOAT32 -> {
                    float[] floatData = getFloatData();
                    Arrays.fill(floatData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value);
                }
                case FLOAT64 -> {
                    double[] doubleData = getDoubleData();
                    Arrays.fill(doubleData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value);
                }
                case BOOLEAN -> {
                    BitSet booleanData = getBooleanData();
                    for (long i = startFlatIndex; i < endFlatIndex; i++) {
                        booleanData.set(Math.toIntExact(i), value != 0);
                    }
                }
            }
        }

        @Override
        public void fillRegion(long startFlatIndex, long endFlatIndex, float value) {
            if (startFlatIndex < 0 || startFlatIndex >= nElements) {
                throw new IndexOutOfBoundsException("Start index " + startFlatIndex + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            if (endFlatIndex < 0 || endFlatIndex > nElements) {
                throw new IndexOutOfBoundsException("End index " + endFlatIndex + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            switch (dataType) {
                case INT8 -> {
                    byte[] byteData = getByteData();
                    Arrays.fill(byteData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), (byte) value);
                }
                case INT16 -> {
                    short[] shortData = getShortData();
                    Arrays.fill(shortData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), (short) value);
                }
                case INT32 -> {
                    int[] intData = getIntData();
                    Arrays.fill(intData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), (int) value);
                }
                case INT64 -> {
                    long[] longData = getLongData();
                    Arrays.fill(longData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), (long) value);
                }
                case FLOAT32 -> {
                    float[] floatData = getFloatData();
                    Arrays.fill(floatData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value);
                }
                case FLOAT64 -> {
                    double[] doubleData = getDoubleData();
                    Arrays.fill(doubleData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value);
                }
                case BOOLEAN -> {
                    BitSet booleanData = getBooleanData();
                    for (long i = startFlatIndex; i < endFlatIndex; i++) {
                        booleanData.set(Math.toIntExact(i), value != 0);
                    }
                }
            }
        }

        @Override
        public void fillRegion(long startFlatIndex, long endFlatIndex, double value) {
            if (startFlatIndex < 0 || startFlatIndex >= nElements) {
                throw new IndexOutOfBoundsException("Start index " + startFlatIndex + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            if (endFlatIndex < 0 || endFlatIndex > nElements) {
                throw new IndexOutOfBoundsException("End index " + endFlatIndex + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            switch (dataType) {
                case INT8 -> {
                    byte[] byteData = getByteData();
                    Arrays.fill(byteData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), (byte) value);
                }
                case INT16 -> {
                    short[] shortData = getShortData();
                    Arrays.fill(shortData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), (short) value);
                }
                case INT32 -> {
                    int[] intData = getIntData();
                    Arrays.fill(intData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), (int) value);
                }
                case INT64 -> {
                    long[] longData = getLongData();
                    Arrays.fill(longData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), (long) value);
                }
                case FLOAT32 -> {
                    float[] floatData = getFloatData();
                    Arrays.fill(floatData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), (float) value);
                }
                case FLOAT64 -> {
                    double[] doubleData = getDoubleData();
                    Arrays.fill(doubleData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value);
                }
                case BOOLEAN -> {
                    BitSet booleanData = getBooleanData();
                    for (long i = startFlatIndex; i < endFlatIndex; i++) {
                        booleanData.set(Math.toIntExact(i), value != 0);
                    }
                }
            }
        }

        @Override
        public void fillRegion(long startFlatIndex, long endFlatIndex, boolean value) {
            if (startFlatIndex < 0 || startFlatIndex >= nElements) {
                throw new IndexOutOfBoundsException("Start index " + startFlatIndex + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            if (endFlatIndex < 0 || endFlatIndex > nElements) {
                throw new IndexOutOfBoundsException("End index " + endFlatIndex + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            switch (dataType) {
                case INT8 -> {
                    byte[] byteData = getByteData();
                    Arrays.fill(byteData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value ? (byte) 1 : (byte) 0);
                }
                case INT16 -> {
                    short[] shortData = getShortData();
                    Arrays.fill(shortData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value ? (short) 1 : (short) 0);
                }
                case INT32 -> {
                    int[] intData = getIntData();
                    Arrays.fill(intData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value ? 1 : 0);
                }
                case INT64 -> {
                    long[] longData = getLongData();
                    Arrays.fill(longData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value ? 1 : 0);
                }
                case FLOAT32 -> {
                    float[] floatData = getFloatData();
                    Arrays.fill(floatData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value ? 1 : 0);
                }
                case FLOAT64 -> {
                    double[] doubleData = getDoubleData();
                    Arrays.fill(doubleData, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex), value ? 1 : 0);
                }
                case BOOLEAN -> {
                    BitSet booleanData = getBooleanData();
                    for (long i = startFlatIndex; i < endFlatIndex; i++) {
                        booleanData.set(Math.toIntExact(i), value);
                    }
                }
            }
        }

        public void setContents(long startIndex, @NotNull BitSet data) {
            if (dataType != DataType.BOOLEAN) {
                throw new UnsupportedOperationException("Cannot set contents of a DataContainer with data type " + dataType);
            }
            BitSet booleanData = getBooleanData();
            if (startIndex < 0 || startIndex >= booleanData.length()) {
                throw new IndexOutOfBoundsException("Start index " + startIndex + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            if (startIndex + data.length() > booleanData.length()) {
                throw new IndexOutOfBoundsException("Buffer size is too large for shape " + ShapeUtils.toString(shape));
            }
            for (int i = 0; i < data.length(); i++) {
                booleanData.set(Math.toIntExact(startIndex + i), data.get(i));
            }
        }

        public void setContents(int startIndex, @NotNull JvmTensorDataContainer container) {
            switch (dataType) {
                case INT8 -> {
                    byte[] newData = container.getByteData();
                    assert byteData != null;
                    System.arraycopy(newData, 0, byteData, startIndex, newData.length);
                }
                case INT16 -> {
                    short[] newData = container.getShortData();
                    assert shortData != null;
                    System.arraycopy(newData, 0, shortData, startIndex, newData.length);
                }
                case INT32 -> {
                    int[] newData = container.getIntData();
                    assert intData != null;
                    System.arraycopy(newData, 0, intData, startIndex, newData.length);
                }
                case INT64 -> {
                    long[] newData = container.getLongData();
                    assert longData != null;
                    System.arraycopy(newData, 0, longData, startIndex, newData.length);
                }
                case FLOAT32 -> {
                    float[] newData = container.getFloatData();
                    assert floatData != null;
                    System.arraycopy(newData, 0, floatData, startIndex, newData.length);
                }
                case FLOAT64 -> {
                    double[] newData = container.getDoubleData();
                    assert doubleData != null;
                    System.arraycopy(newData, 0, doubleData, startIndex, newData.length);
                }
            }
        }

        /**
         * Returns the contents in the specified index interval of the tensor as a direct buffer. The buffer must be freed by the caller via JEmalloc.je_free().
         *
         * @param startFlatIndex The start index of the interval.
         * @param endFlatIndex   The end index of the interval. (exclusive)
         * @return the direct buffer with tensor contents
         */
        @NotNull
        public DirectMemoryHandle getAsDirectBuffer(long startFlatIndex, long endFlatIndex) {
            DirectMemoryHandle memoryHandle = backend.getDirectMemoryManager().alloc(endFlatIndex - startFlatIndex, dataType);
            switch (dataType) {
                case INT8 -> {
                    byte[] data = getByteData();
                    if (startFlatIndex < 0 || endFlatIndex > data.length) {
                        throw new IllegalArgumentException("Index out of bounds: " + startFlatIndex + " to " + endFlatIndex + " (length " + data.length + ")");
                    }
                    memoryHandle.asByteBuffer().put(data, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex - startFlatIndex));
                }
                case INT16 -> {
                    short[] data = getShortData();
                    if (startFlatIndex < 0 || endFlatIndex > data.length) {
                        throw new IllegalArgumentException("Index out of bounds: " + startFlatIndex + " to " + endFlatIndex + " (length " + data.length + ")");
                    }
                    memoryHandle.asShortBuffer().put(data, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex - startFlatIndex));
                }
                case INT32 -> {
                    int[] data = getIntData();
                    if (startFlatIndex < 0 || endFlatIndex > data.length) {
                        throw new IllegalArgumentException("Index out of bounds: " + startFlatIndex + " to " + endFlatIndex + " (length " + data.length + ")");
                    }
                    memoryHandle.asIntBuffer().put(data, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex - startFlatIndex));
                }
                case INT64 -> {
                    long[] data = getLongData();
                    if (startFlatIndex < 0 || endFlatIndex > data.length) {
                        throw new IllegalArgumentException("Index out of bounds: " + startFlatIndex + " to " + endFlatIndex + " (length " + data.length + ")");
                    }
                    memoryHandle.asLongBuffer().put(data, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex - startFlatIndex));
                }
                case FLOAT32 -> {
                    float[] data = getFloatData();
                    if (startFlatIndex < 0 || endFlatIndex > data.length) {
                        throw new IllegalArgumentException("Index out of bounds: " + startFlatIndex + " to " + endFlatIndex + " (length " + data.length + ")");
                    }
                    memoryHandle.asFloatBuffer().put(data, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex - startFlatIndex));
                }
                case FLOAT64 -> {
                    double[] data = getDoubleData();
                    if (startFlatIndex < 0 || endFlatIndex > data.length) {
                        throw new IllegalArgumentException("Index out of bounds: " + startFlatIndex + " to " + endFlatIndex + " (length " + data.length + ")");
                    }
                    memoryHandle.asDoubleBuffer().put(data, Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex - startFlatIndex));
                }
                case BOOLEAN -> {
                    BitSet data = getBooleanData();
                    ByteBuffer buffer = memoryHandle.asByteBuffer();
                    if (startFlatIndex < 0 || endFlatIndex > data.length()) {
                        throw new IllegalArgumentException("Index out of bounds: " + startFlatIndex + " to " + endFlatIndex + " (length " + data.length() + ")");
                    }
                    for (int i = Math.toIntExact(startFlatIndex); i < endFlatIndex; i++) {
                        int byteIndex = i / 8;
                        int bitIndex = i % 8;
                        if (data.get(i)) {
                            buffer.put(byteIndex, (byte) (buffer.get(byteIndex) | (1 << bitIndex)));
                        }
                    }
                }
                default -> throw new UnsupportedOperationException("Unsupported data type " + dataType);
            }
            return memoryHandle;
        }

        @Override
        public @NotNull DirectMemoryHandle getAsDirectBuffer() {
            return getAsDirectBuffer(0, nElements);
        }

        @Override
        public void dispose() {
            this.bitSetData = null;
            this.byteData = null;
            this.shortData = null;
            this.intData = null;
            this.longData = null;
            this.floatData = null;
            this.doubleData = null;
        }
    }

    @Override
    public @NotNull DirectMemoryHandle getContentsAsDirectMemory() {
        long nElements = getNumberOfElements();
        if (nElements > Integer.MAX_VALUE) {
            throw new UnsupportedOperationException("JvmTensors cannot have more than Integer.MAX_VALUE elements");
        }
        return this.dataContainer.getAsDirectBuffer(0, Math.toIntExact(nElements));
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
    public void readFrom(@NotNull InputStream inputStream) throws IOException {
        try (DataInputStream dataInputStream = new DataInputStream(inputStream)) {
            switch (getDataType()) {
                case INT8 -> {
                    byte[] byteData = this.dataContainer.getByteData();
                    dataInputStream.readFully(byteData);
                }
                case INT16 -> {
                    short[] shortData = this.dataContainer.getShortData();
                    for (int i = 0; i < shortData.length; i++) {
                        shortData[i] = dataInputStream.readShort();
                    }
                }
                case INT32 -> {
                    int[] intData = this.dataContainer.getIntData();
                    for (int i = 0; i < intData.length; i++) {
                        intData[i] = dataInputStream.readInt();
                    }
                }
                case INT64 -> {
                    long[] longData = this.dataContainer.getLongData();
                    for (int i = 0; i < longData.length; i++) {
                        longData[i] = dataInputStream.readLong();
                    }
                }
                case FLOAT32 -> {
                    float[] floatData = this.dataContainer.getFloatData();
                    for (int i = 0; i < floatData.length; i++) {
                        floatData[i] = dataInputStream.readFloat();
                    }
                }
                case FLOAT64 -> {
                    double[] doubleData = this.dataContainer.getDoubleData();
                    for (int i = 0; i < doubleData.length; i++) {
                        doubleData[i] = dataInputStream.readDouble();
                    }
                }
                case BOOLEAN -> {
                    BitSet bitSetData = this.dataContainer.getBooleanData();
                    byte currentByte = 0;
                    for (int i = 0; i < bitSetData.length(); i++) {
                        if (i % 8 == 0) {
                            currentByte = dataInputStream.readByte();
                        }
                        bitSetData.set(i, (currentByte & (1 << (i % 8))) != 0);
                    }
                }
            }
        }
    }

    @Override
    public void dispose() {
        super.dispose();
        this.dataContainer.dispose();
    }

    @Override
    public @NotNull ISciCoreBackend getSciCoreBackend() {
        return this.backend;
    }
}
