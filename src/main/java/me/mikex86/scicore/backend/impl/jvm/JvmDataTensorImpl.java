package me.mikex86.scicore.backend.impl.jvm;

import me.mikex86.scicore.*;
import me.mikex86.scicore.backend.ITensorImpl;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;
import java.util.Objects;

public class JvmDataTensorImpl implements ITensorImpl {

    @NotNull
    private final JvmTensorDataContainer dataContainer;

    private final long @NotNull [] strides;

    @NotNull
    private final ISciCoreBackend backend;


    public JvmDataTensorImpl(@NotNull ISciCoreBackend backend, @NotNull DataType dataType, long @NotNull [] shape) {
        this.backend = backend;
        this.dataContainer = new JvmTensorDataContainer(dataType, shape);
        this.strides = ShapeUtils.makeStrides(shape);
    }

    JvmDataTensorImpl(@NotNull ISciCoreBackend backend, @NotNull JvmTensorDataContainer dataContainer, long @NotNull [] shape) {
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
    public byte getByte(long @NotNull [] indices) {
        long index = ShapeUtils.getFlatIndex(indices, this.strides);
        return this.dataContainer.getByte(index);
    }

    @Override
    public void setByte(byte value, long @NotNull [] indices) {
        long index = ShapeUtils.getFlatIndex(indices, this.strides);
        this.dataContainer.setByte(index, value);
    }

    @Override
    public short getShort(long @NotNull [] indices) {
        long index = ShapeUtils.getFlatIndex(indices, this.strides);
        return this.dataContainer.getShort(index);
    }

    @Override
    public void setShort(short value, long @NotNull [] indices) {
        long index = ShapeUtils.getFlatIndex(indices, this.strides);
        this.dataContainer.setShort(index, value);
    }

    @Override
    public int getInt(long @NotNull [] indices) {
        long index = ShapeUtils.getFlatIndex(indices, this.strides);
        return this.dataContainer.getInt(index);
    }

    @Override
    public void setInt(int value, long @NotNull [] indices) {
        long index = ShapeUtils.getFlatIndex(indices, this.strides);
        this.dataContainer.setInt(index, value);
    }

    @Override
    public long getLong(long @NotNull [] indices) {
        long index = ShapeUtils.getFlatIndex(indices, this.strides);
        return this.dataContainer.getLong(index);
    }

    @Override
    public void setLong(long value, long @NotNull [] indices) {
        long index = ShapeUtils.getFlatIndex(indices, this.strides);
        this.dataContainer.setLong(index, value);
    }

    @Override
    public float getFloat(long @NotNull [] indices) {
        long index = ShapeUtils.getFlatIndex(indices, this.strides);
        return this.dataContainer.getFloat(index);
    }

    @Override
    public void setFloat(float value, long @NotNull [] indices) {
        long index = ShapeUtils.getFlatIndex(indices, this.strides);
        this.dataContainer.setFloat(index, value);
    }

    @Override
    public double getDouble(long @NotNull [] indices) {
        long index = ShapeUtils.getFlatIndex(indices, this.strides);
        return this.dataContainer.getDouble(index);
    }

    @Override
    public void setDouble(double value, long @NotNull [] indices) {
        long index = ShapeUtils.getFlatIndex(indices, this.strides);
        this.dataContainer.setDouble(index, value);
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
    public @NotNull ITensorImpl copy() {
        ITensorImpl copy = new JvmDataTensorImpl(this.backend, getDataType(), getShape());
        copy.setContents(this);
        return copy;
    }

    @Override
    public void setContents(@NotNull ITensor tensor) {
        if (tensor instanceof Tensor t && t.getTensorImpl() instanceof JvmDataTensorImpl jvmTensorImpl) {
            this.dataContainer.setContents(jvmTensorImpl.dataContainer);
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
    public void setContents(long @NotNull [] dimension, @NotNull ITensor tensor, boolean useView) {
        throw new UnsupportedOperationException("TODO"); // TODO: IMPLEMENT
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

        private byte getByte(long index) {
            byte[] byteData = getByteData();
            if (index < 0 || index >= byteData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            return byteData[(int) index];
        }

        private short getShort(long index) {
            short[] shortData = getShortData();
            if (index < 0 || index >= shortData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            return shortData[(int) index];
        }

        private int getInt(long index) {
            int[] intData = getIntData();
            if (index < 0 || index >= intData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            return intData[(int) index];
        }

        private long getLong(long index) {
            long[] longData = getLongData();
            if (index < 0 || index >= longData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            return longData[(int) index];
        }

        private float getFloat(long index) {
            float[] floatData = getFloatData();
            if (index < 0 || index >= floatData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            return floatData[(int) index];
        }

        private double getDouble(long index) {
            double[] doubleData = getDoubleData();
            if (index < 0 || index >= doubleData.length) {
                throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(shape));
            }
            return doubleData[(int) index];
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

        @NotNull
        public DataType getDataType() {
            return dataType;
        }

        public long @NotNull [] getShape() {
            return shape;
        }

        @NotNull
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
    }

    @Override
    public String toString() {
        long[] shape = getShape();
        StringBuilder sb = new StringBuilder("JvmTensor(dtype=" + getDataType() + ", shape=" + Arrays.toString(shape) + ", data=\n");
        ITensorIterator iterator = iterator();
        boolean isNewLine = true;
        int nElementsInLine = 0;
        while (iterator.hasNext()) {
            long nStartingDimensions = iterator.getNumStartingDimensions();
            long nEndingDimensions = iterator.getNumEndingDimensions();
            if (isNewLine) {
                sb.append("\t");
            }
            if (isNewLine) {
                for (int i = 0; i < shape.length - nStartingDimensions; i++) {
                    sb.append(" ");
                }
            }
            for (long i = 0; i < nStartingDimensions; i++) {
                sb.append("[");
            }
            switch (iterator.getDataType()) {
                case INT8 -> sb.append(iterator.getByte());
                case INT16 -> sb.append(iterator.getShort());
                case INT32 -> sb.append(iterator.getInt());
                case INT64 -> sb.append(iterator.getLong());
                case FLOAT32 -> sb.append(String.format("%4.3g", iterator.getFloat()));
                case FLOAT64 -> sb.append(String.format("%4.3g", iterator.getDouble()));
            }
            for (long i = 0; i < nEndingDimensions; i++) {
                sb.append("]");
            }
            iterator.moveNext();
            if (!iterator.hasNext()) {
                continue;
            }
            sb.append(",");
            //if ((nElementsInLine++ >= 5 && nEndingDimensions > 0) || nElementsInLine >= 10) {
            if (nEndingDimensions > 0) {
                sb.append("\n");
                isNewLine = true;
                nElementsInLine = 0;
            } else {
                sb.append(" ");
                isNewLine = false;
            }
        }
        sb.append(")\n");
        return sb.toString();
    }

    @Override
    public @NotNull
    ITensorIterator iterator() {
        return new DefaultTensorIterator(this);
    }

    @Override
    public @NotNull ISciCoreBackend getSciCoreBackend() {
        return this.backend;
    }
}
