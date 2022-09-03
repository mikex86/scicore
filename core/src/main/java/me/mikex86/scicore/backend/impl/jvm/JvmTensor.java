package me.mikex86.scicore.backend.impl.jvm;

import me.mikex86.scicore.*;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.op.IGraphRecorder;
import me.mikex86.scicore.utils.Pair;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.lwjgl.system.jemalloc.JEmalloc;

import java.nio.*;
import java.util.BitSet;
import java.util.Objects;

public class JvmTensor extends AbstractTensor implements ITensor {

    @NotNull
    private final JvmTensorDataContainer dataContainer;

    private final long @NotNull [] strides;

    @NotNull
    private final ISciCoreBackend backend;


    public JvmTensor(@NotNull ISciCoreBackend backend, @NotNull DataType dataType, long @NotNull [] shape) {
        this.numElements = ShapeUtils.getNumElements(shape);
        this.dataContainer = new JvmTensorDataContainer(dataType, shape);
        this.strides = ShapeUtils.makeStrides(shape);
        this.backend = backend;
    }

    JvmTensor(@NotNull ISciCoreBackend backend,@NotNull JvmTensorDataContainer dataContainer, long @NotNull [] shape) {
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
    public short getShort(long @NotNull [] indices) {
        long index = ShapeUtils.getFlatIndex(indices, this.strides);
        return this.dataContainer.getShort(index);
    }

    @Override
    public int getInt(long @NotNull [] indices) {
        long index = ShapeUtils.getFlatIndex(indices, this.strides);
        return this.dataContainer.getInt(index);
    }

    @Override
    public long getLong(long @NotNull [] indices) {
        long index = ShapeUtils.getFlatIndex(indices, this.strides);
        return this.dataContainer.getLong(index);
    }

    @Override
    public boolean getBooleanFlat(long flatIndex) {
        return this.dataContainer.getBoolean(flatIndex);
    }

    @Override
    public void setBooleanFlat(boolean value, long flatIndex) {
        this.dataContainer.setBoolean(flatIndex, value);
    }

    @Override
    public byte getByteFlat(long flatIndex) {
        return this.dataContainer.getByte(flatIndex);
    }

    @Override
    public void setByteFlat(byte value, long flatIndex) {
        this.dataContainer.setByte(flatIndex, value);
    }

    @Override
    public short getShortFlat(long flatIndex) {
        return this.dataContainer.getShort(flatIndex);
    }

    @Override
    public void setShortFlat(short value, long flatIndex) {
        this.dataContainer.setShort(flatIndex, value);
    }

    @Override
    public int getIntFlat(long flatIndex) {
        return this.dataContainer.getInt(flatIndex);
    }

    @Override
    public void setIntFlat(int value, long flatIndex) {
        this.dataContainer.setInt(flatIndex, value);
    }

    @Override
    public long getLongFlat(long flatIndex) {
        return this.dataContainer.getLong(flatIndex);
    }

    @Override
    public void setLongFlat(long value, long flatIndex) {
        this.dataContainer.setLong(flatIndex, value);
    }

    @Override
    public float getFloatFlat(long flatIndex) {
        return this.dataContainer.getFloat(flatIndex);
    }

    @Override
    public void setFloatFlat(float value, long flatIndex) {
        this.dataContainer.setFloat(flatIndex, value);
    }

    @Override
    public double getDoubleFlat(long flatIndex) {
        return this.dataContainer.getDouble(flatIndex);
    }

    @Override
    public void setDoubleFlat(double value, long flatIndex) {
        this.dataContainer.setDouble(flatIndex, value);
    }

    @Override
    public @NotNull ITensor copy() {
        ITensor copy = new JvmTensor(backend, getDataType(), getShape());
        copy.setContents(this);
        return copy;
    }

    @Override
    public void setContents(@NotNull ITensor tensor) {
        if (tensor instanceof JvmTensor jvmTensor) {
            this.dataContainer.setContents(jvmTensor.dataContainer);
        } else {
            // General copy
            long nElements = tensor.getNumberOfElements();
            for (long i = 0; i < nElements; i++) {
                switch (this.getDataType()) {
                    case INT8 -> this.setByteFlat(tensor.getByteFlat(i), i);
                    case INT16 -> this.setShortFlat(tensor.getShortFlat(i), i);
                    case INT32 -> this.setIntFlat(tensor.getIntFlat(i), i);
                    case INT64 -> this.setLongFlat(tensor.getLongFlat(i), i);
                    case FLOAT32 -> this.setFloatFlat(tensor.getFloatFlat(i), i);
                    case FLOAT64 -> this.setDoubleFlat(tensor.getDoubleFlat(i), i);
                    default -> throw new IllegalArgumentException("Unsupported data type");
                }
            }
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
                case INT8 -> this.setByteFlat(tensor.getByteFlat(i), startIndex + i);
                case INT16 -> this.setShortFlat(tensor.getShortFlat(i), startIndex + i);
                case INT32 -> this.setIntFlat(tensor.getIntFlat(i), startIndex + i);
                case INT64 -> this.setLongFlat(tensor.getLongFlat(i), startIndex + i);
                case FLOAT32 -> this.setFloatFlat(tensor.getFloatFlat(i), startIndex + i);
                case FLOAT64 -> this.setDoubleFlat(tensor.getDoubleFlat(i), startIndex + i);
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
                    dataContainer.setByte(j, i);
                }
            }
            case INT16 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setShort(j, i);
                }
            }
            case INT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt(j, i);
                }
            }
            case INT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setLong(j, i);
                }
            }
            case FLOAT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setFloat(j, i);
                }
            }
            case FLOAT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setDouble(j, i);
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
                    dataContainer.setByte(j, (byte) i);
                }
            }
            case INT16 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setShort(j, i);
                }
            }
            case INT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt(j, i);
                }
            }
            case INT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setLong(j, i);
                }
            }
            case FLOAT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setFloat(j, i);
                }
            }
            case FLOAT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setDouble(j, i);
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
                    dataContainer.setByte(j, (byte) i);
                }
            }
            case INT16 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setShort(j, (short) i);
                }
            }
            case INT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt(j, i);
                }
            }
            case INT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setLong(j, i);
                }
            }
            case FLOAT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setFloat(j, i);
                }
            }
            case FLOAT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setDouble(j, i);
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
                    dataContainer.setByte(j, (byte) i);
                }
            }
            case INT16 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setShort(j, (short) i);
                }
            }
            case INT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt(j, (int) i);
                }
            }
            case INT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setLong(j, i);
                }
            }
            case FLOAT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setFloat(j, (float) i);
                }
            }
            case FLOAT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setDouble(j, i);
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
                    dataContainer.setByte(j, (byte) f);
                }
            }
            case INT16 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setShort(j, (short) f);
                }
            }
            case INT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt(j, (int) f);
                }
            }
            case INT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setLong(j, (long) f);
                }
            }
            case FLOAT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setFloat(j, f);
                }
            }
            case FLOAT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setDouble(j, f);
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
                    dataContainer.setByte(j, (byte) d);
                }
            }
            case INT16 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setShort(j, (short) d);
                }
            }
            case INT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt(j, (int) d);
                }
            }
            case INT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setLong(j, (long) d);
                }
            }
            case FLOAT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setFloat(j, (float) d);
                }
            }
            case FLOAT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setDouble(j, d);
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
                    dataContainer.setByte(j, (byte) (value ? 1 : 0));
                }
            }
            case INT16 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setShort(j, (short) (value ? 1 : 0));
                }
            }
            case INT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setInt(j, value ? 1 : 0);
                }
            }
            case INT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setLong(j, value ? 1 : 0);
                }
            }
            case FLOAT32 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setFloat(j, value ? 1 : 0);
                }
            }
            case FLOAT64 -> {
                for (long j = 0; j < nElements; j++) {
                    dataContainer.setDouble(j, value ? 1 : 0);
                }
            }
        }
    }


    private static class JvmTensorDataContainer {

        @NotNull
        private final DataType dataType;

        private final long @NotNull [] shape;

        private byte @Nullable [] byteData;
        private short @Nullable [] shortData;
        private int @Nullable [] intData;
        private long @Nullable [] longData;
        private float @Nullable [] floatData;
        private double @Nullable [] doubleData;

        private @Nullable BitSet bitSetData;
        private int nBits;

        private JvmTensorDataContainer(@NotNull DataType dataType, long @NotNull [] shape) {
            this.dataType = dataType;
            this.shape = shape;
            switch (dataType) {
                case INT8 -> this.byteData = new byte[Math.toIntExact(ShapeUtils.getNumElements(shape))];
                case INT16 -> this.shortData = new short[Math.toIntExact(ShapeUtils.getNumElements(shape))];
                case INT32 -> this.intData = new int[Math.toIntExact(ShapeUtils.getNumElements(shape))];
                case INT64 -> this.longData = new long[Math.toIntExact(ShapeUtils.getNumElements(shape))];
                case FLOAT32 -> this.floatData = new float[Math.toIntExact(ShapeUtils.getNumElements(shape))];
                case FLOAT64 -> this.doubleData = new double[Math.toIntExact(ShapeUtils.getNumElements(shape))];
                case BOOLEAN -> {
                    this.nBits = Math.toIntExact(ShapeUtils.getNumElements(shape));
                    this.bitSetData = new BitSet(nBits);
                }
            }
        }

        private JvmTensorDataContainer(long @NotNull [] shape, byte @NotNull [] byteData) {
            this.dataType = DataType.INT8;
            this.shape = shape;
            this.byteData = byteData;
        }

        private JvmTensorDataContainer(long @NotNull [] shape, short @NotNull [] shortData) {
            this.dataType = DataType.INT16;
            this.shape = shape;
            this.shortData = shortData;
        }

        private JvmTensorDataContainer(long @NotNull [] shape, int @NotNull [] intData) {
            this.dataType = DataType.INT32;
            this.shape = shape;
            this.intData = intData;
        }

        private JvmTensorDataContainer(long @NotNull [] shape, long @NotNull [] longData) {
            this.dataType = DataType.INT64;
            this.shape = shape;
            this.longData = longData;
        }

        private JvmTensorDataContainer(long @NotNull [] shape, float @NotNull [] floatData) {
            this.dataType = DataType.FLOAT32;
            this.shape = shape;
            this.floatData = floatData;
        }

        private JvmTensorDataContainer(long @NotNull [] shape, double @NotNull [] doubleData) {
            this.dataType = DataType.FLOAT64;
            this.shape = shape;
            this.doubleData = doubleData;
        }

        private byte @NotNull [] getByteData() {
            return Objects.requireNonNull(byteData, "DataContainer has different data type");
        }

        private short @NotNull [] getShortData() {
            return Objects.requireNonNull(shortData, "DataContainer has different data type");
        }

        private int @NotNull [] getIntData() {
            return Objects.requireNonNull(intData, "DataContainer has different data type");
        }

        private float @NotNull [] getFloatData() {
            return Objects.requireNonNull(floatData, "DataContainer has different data type");
        }

        private double @NotNull [] getDoubleData() {
            return Objects.requireNonNull(doubleData, "DataContainer has different data type");
        }

        private long @NotNull [] getLongData() {
            return Objects.requireNonNull(longData, "DataContainer has different data type");
        }

        private @NotNull BitSet getBooleanData() {
            return Objects.requireNonNull(bitSetData, "DataContainer has different data type");
        }

        public byte getByte(long index) {
            byte[] byteData = getByteData();
            if (index < 0 || index >= byteData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            return byteData[(int) index];
        }

        public short getShort(long index) {
            short[] shortData = getShortData();
            if (index < 0 || index >= shortData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            return shortData[(int) index];
        }

        public int getInt(long index) {
            int[] intData = getIntData();
            if (index < 0 || index >= intData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            return intData[(int) index];
        }

        public long getLong(long index) {
            long[] longData = getLongData();
            if (index < 0 || index >= longData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            return longData[(int) index];
        }

        public float getFloat(long index) {
            float[] floatData = getFloatData();
            if (index < 0 || index >= floatData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            return floatData[(int) index];
        }

        public double getDouble(long index) {
            double[] doubleData = getDoubleData();
            if (index < 0 || index >= doubleData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            return doubleData[(int) index];
        }

        public boolean getBoolean(long index) {
            BitSet bitSetData = getBooleanData();
            if (index < 0 || index >= nBits) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            return bitSetData.get((int) index);
        }

        public void setByte(long index, byte value) {
            byte[] byteData = getByteData();
            if (index < 0 || index >= byteData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            byteData[(int) index] = value;
        }

        public void setShort(long index, short value) {
            short[] shortData = getShortData();
            if (index < 0 || index >= shortData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            shortData[(int) index] = value;
        }

        public void setInt(long index, int value) {
            int[] intData = getIntData();
            if (index < 0 || index >= intData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            intData[(int) index] = value;
        }

        public void setLong(long index, long value) {
            long[] longData = getLongData();
            if (index < 0 || index >= longData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            longData[(int) index] = value;
        }

        public void setFloat(long index, float value) {
            float[] floatData = getFloatData();
            if (index < 0 || index >= floatData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            floatData[(int) index] = value;
        }

        public void setDouble(long index, double value) {
            double[] doubleData = getDoubleData();
            if (index < 0 || index >= doubleData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            doubleData[(int) index] = value;
        }

        public void setBoolean(long index, boolean value) {
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

        public long @NotNull [] getShape() {
            return shape;
        }

        public void setContents(@NotNull ByteBuffer buffer) {
            if (dataType != DataType.INT8) {
                throw new UnsupportedOperationException("Cannot set contents of a DataContainer with data type " + dataType);
            }
            byte[] byteData = getByteData();
            if (buffer.remaining() > byteData.length) {
                throw new IllegalArgumentException("Contents too large: Buffer has " + buffer.remaining() + " bytes, but DataContainer has " + byteData.length + " bytes");
            }
            buffer.get(byteData);
        }

        public void setContents(@NotNull ShortBuffer buffer) {
            if (dataType != DataType.INT16) {
                throw new UnsupportedOperationException("Cannot set contents of a DataContainer with data type " + dataType);
            }
            short[] shortData = getShortData();
            if (buffer.remaining() > shortData.length) {
                throw new IllegalArgumentException("Contents too large: Buffer has " + buffer.remaining() + " shorts, but DataContainer has " + shortData.length + " shorts");
            }
            buffer.get(shortData);
        }

        public void setContents(@NotNull IntBuffer buffer) {
            if (dataType != DataType.INT32) {
                throw new UnsupportedOperationException("Cannot set contents of a DataContainer with data type " + dataType);
            }
            int[] intData = getIntData();
            if (buffer.remaining() > intData.length) {
                throw new IllegalArgumentException("Contents too large: Buffer has " + buffer.remaining() + " ints, but DataContainer has " + intData.length + " ints");
            }
            buffer.get(intData);
        }

        public void setContents(@NotNull LongBuffer buffer) {
            if (dataType != DataType.INT64) {
                throw new UnsupportedOperationException("Cannot set contents of a DataContainer with data type " + dataType);
            }
            long[] longData = getLongData();
            if (buffer.remaining() > longData.length) {
                throw new IllegalArgumentException("Contents too large: Buffer has " + buffer.remaining() + " longs, but DataContainer has " + longData.length + " longs");
            }
            buffer.get(longData);
        }

        public void setContents(@NotNull FloatBuffer buffer) {
            if (dataType != DataType.FLOAT32) {
                throw new UnsupportedOperationException("Cannot set contents of a DataContainer with data type " + dataType);
            }
            float[] floatData = getFloatData();
            if (buffer.remaining() > floatData.length) {
                throw new IllegalArgumentException("Contents too large: Buffer has " + buffer.remaining() + " floats, but DataContainer has " + floatData.length + " floats");
            }
            buffer.get(floatData);
        }

        public void setContents(@NotNull DoubleBuffer buffer) {
            if (dataType != DataType.FLOAT64) {
                throw new UnsupportedOperationException("Cannot set contents of a DataContainer with data type " + dataType);
            }
            double[] doubleData = getDoubleData();
            if (buffer.remaining() > doubleData.length) {
                throw new IllegalArgumentException("Contents too large: Buffer has " + buffer.remaining() + " doubles, but DataContainer has " + doubleData.length + " doubles");
            }
            buffer.get(doubleData);
        }

        public void setContents(boolean @NotNull [] data) {
            if (dataType != DataType.BOOLEAN) {
                throw new UnsupportedOperationException("Cannot set contents of a DataContainer with data type " + dataType);
            }
            BitSet booleanData = getBooleanData();
            if (data.length > nBits) {
                throw new IllegalArgumentException("Contents too large: Array has " + data.length + " booleans, but DataContainer has " + nBits + " booleans");
            }
            for (int i = 0; i < data.length; i++) {
                booleanData.set(i, data[i]);
            }
        }

        public void setContents(@NotNull BitSet data) {
            if (dataType != DataType.BOOLEAN) {
                throw new UnsupportedOperationException("Cannot set contents of a DataContainer with data type " + dataType);
            }
            BitSet booleanData = getBooleanData();
            if (data.length() > nBits) {
                throw new IllegalArgumentException("Contents too large: BitSet has " + data.length() + " booleans, but DataContainer has " + nBits + " booleans");
            }
            booleanData.clear();
            booleanData.or(data);
        }

        public void setContents(@NotNull JvmTensorDataContainer container) {
            switch (dataType) {
                case INT8 -> {
                    byte[] newData = container.getByteData();
                    assert byteData != null;
                    if (byteData.length != newData.length) {
                        byteData = new byte[newData.length];
                    }
                    System.arraycopy(newData, 0, byteData, 0, newData.length);
                }
                case INT16 -> {
                    short[] newData = container.getShortData();
                    assert shortData != null;
                    if (shortData.length != newData.length) {
                        shortData = new short[newData.length];
                    }
                    System.arraycopy(newData, 0, shortData, 0, newData.length);
                }
                case INT32 -> {
                    int[] newData = container.getIntData();
                    assert intData != null;
                    if (intData.length != newData.length) {
                        intData = new int[newData.length];
                    }
                    System.arraycopy(newData, 0, intData, 0, newData.length);
                }
                case INT64 -> {
                    long[] newData = container.getLongData();
                    assert longData != null;
                    if (longData.length != newData.length) {
                        longData = new long[newData.length];
                    }
                    System.arraycopy(newData, 0, longData, 0, newData.length);
                }
                case FLOAT32 -> {
                    float[] newData = container.getFloatData();
                    assert floatData != null;
                    if (floatData.length != newData.length) {
                        floatData = new float[newData.length];
                    }
                    System.arraycopy(newData, 0, floatData, 0, newData.length);
                }
                case FLOAT64 -> {
                    double[] newData = container.getDoubleData();
                    assert doubleData != null;
                    if (doubleData.length != newData.length) {
                        doubleData = new double[newData.length];
                    }
                    System.arraycopy(newData, 0, doubleData, 0, newData.length);
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
        public ByteBuffer getAsDirectBuffer(int startFlatIndex, int endFlatIndex) {
            ByteBuffer buffer;
            switch (dataType) {
                case INT8 -> {
                    byte[] data = getByteData();
                    if (startFlatIndex < 0 || endFlatIndex > data.length) {
                        throw new IllegalArgumentException("Index out of bounds: " + startFlatIndex + " to " + endFlatIndex + " (length " + data.length + ")");
                    }
                    buffer = JEmalloc.je_malloc((long) (endFlatIndex - startFlatIndex) * Byte.BYTES);
                    if (buffer == null) {
                        throw new OutOfMemoryError("Could not allocate " + data.length + " bytes");
                    }
                    buffer.put(data, startFlatIndex, endFlatIndex - startFlatIndex);
                }
                case INT16 -> {
                    short[] data = getShortData();
                    if (startFlatIndex < 0 || endFlatIndex > data.length) {
                        throw new IllegalArgumentException("Index out of bounds: " + startFlatIndex + " to " + endFlatIndex + " (length " + data.length + ")");
                    }
                    buffer = JEmalloc.je_malloc((long) (endFlatIndex - startFlatIndex) * Short.BYTES);
                    if (buffer == null) {
                        throw new OutOfMemoryError("Could not allocate " + data.length + " bytes");
                    }
                    buffer.asShortBuffer().put(data, startFlatIndex, endFlatIndex - startFlatIndex);
                }
                case INT32 -> {
                    int[] data = getIntData();
                    if (startFlatIndex < 0 || endFlatIndex > data.length) {
                        throw new IllegalArgumentException("Index out of bounds: " + startFlatIndex + " to " + endFlatIndex + " (length " + data.length + ")");
                    }
                    buffer = JEmalloc.je_malloc((long) (endFlatIndex - startFlatIndex) * Integer.BYTES);
                    if (buffer == null) {
                        throw new OutOfMemoryError("Could not allocate " + data.length + " bytes");
                    }
                    buffer.asIntBuffer().put(data, startFlatIndex, endFlatIndex - startFlatIndex);
                }
                case INT64 -> {
                    long[] data = getLongData();
                    if (startFlatIndex < 0 || endFlatIndex > data.length) {
                        throw new IllegalArgumentException("Index out of bounds: " + startFlatIndex + " to " + endFlatIndex + " (length " + data.length + ")");
                    }
                    buffer = JEmalloc.je_malloc((long) (endFlatIndex - startFlatIndex) * Long.BYTES);
                    if (buffer == null) {
                        throw new OutOfMemoryError("Could not allocate " + data.length + " bytes");
                    }
                    buffer.asLongBuffer().put(data, startFlatIndex, endFlatIndex - startFlatIndex);
                }
                case FLOAT32 -> {
                    float[] data = getFloatData();
                    if (startFlatIndex < 0 || endFlatIndex > data.length) {
                        throw new IllegalArgumentException("Index out of bounds: " + startFlatIndex + " to " + endFlatIndex + " (length " + data.length + ")");
                    }
                    buffer = JEmalloc.je_malloc((long) (endFlatIndex - startFlatIndex) * Float.BYTES);
                    if (buffer == null) {
                        throw new OutOfMemoryError("Could not allocate " + data.length + " bytes");
                    }
                    buffer.asFloatBuffer().put(data, startFlatIndex, endFlatIndex - startFlatIndex);
                }
                case FLOAT64 -> {
                    double[] data = getDoubleData();
                    if (startFlatIndex < 0 || endFlatIndex > data.length) {
                        throw new IllegalArgumentException("Index out of bounds: " + startFlatIndex + " to " + endFlatIndex + " (length " + data.length + ")");
                    }
                    buffer = JEmalloc.je_malloc((long) (endFlatIndex - startFlatIndex) * Double.BYTES);
                    if (buffer == null) {
                        throw new OutOfMemoryError("Could not allocate " + data.length + " bytes");
                    }
                    buffer.asDoubleBuffer().put(data, startFlatIndex, endFlatIndex - startFlatIndex);
                }
                case BOOLEAN -> {
                    BitSet data = getBooleanData();
                    int nBytes = (nBits + 7) / 8;
                    buffer = JEmalloc.je_malloc(nBytes);
                    if (buffer == null) {
                        throw new OutOfMemoryError("Could not allocate " + nBytes + " bytes");
                    }
                    for (int i = 0; i < nBits; i++) {
                        int byteIndex = i / 8;
                        int bitIndex = i % 8;
                        if (data.get(i)) {
                            buffer.put(byteIndex, (byte) (buffer.get(byteIndex) | (1 << bitIndex)));
                        }
                    }
                }
                default -> throw new UnsupportedOperationException("Unsupported data type " + dataType);
            }
            buffer.flip();
            return buffer;
        }
    }

    @Override
    public @NotNull Pair<ByteBuffer, Boolean> getAsDirectBuffer() {
        long nElements = getNumberOfElements();
        if (nElements > Integer.MAX_VALUE) {
            throw new UnsupportedOperationException("JvmTensors cannot have more than Integer.MAX_VALUE elements");
        }
        return Pair.of(this.dataContainer.getAsDirectBuffer(0, Math.toIntExact(nElements)), true);
    }


    @Override
    public @NotNull Pair<ByteBuffer, Boolean> getAsDirectBuffer(long startFlatIndex, long endFlatIndex) {
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
        return Pair.of(this.dataContainer.getAsDirectBuffer(Math.toIntExact(startFlatIndex), Math.toIntExact(endFlatIndex)), true);
    }

    @Override
    public @NotNull ITensorIterator iterator() {
        return new DefaultTensorIterator(this);
    }

    @Override
    public @NotNull ISciCoreBackend getSciCoreBackend() {
        return this.backend;
    }
}
