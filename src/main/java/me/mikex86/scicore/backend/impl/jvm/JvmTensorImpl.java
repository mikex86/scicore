package me.mikex86.scicore.backend.impl.jvm;

import me.mikex86.scicore.*;
import me.mikex86.scicore.backend.TensorImpl;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;
import java.util.Objects;

public class JvmTensorImpl implements TensorImpl {

    @NotNull
    private final JvmTensorDataContainer dataContainer;

    private final long @NotNull [] strides;

    public JvmTensorImpl(@NotNull DataType dataType, long @NotNull [] shape) {
        this.dataContainer = new JvmTensorDataContainer(dataType, shape);
        this.strides = ShapeUtils.makeStrides(shape);
    }

    JvmTensorImpl(@NotNull JvmTensorDataContainer dataContainer, long @NotNull [] shape) {
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
    public void setContents(@NotNull ITensor tensor) {
        if (tensor instanceof Tensor t && t.getTensorImpl() instanceof JvmTensorImpl jvmTensorImpl) {
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
    public @NotNull
    JvmTensorImpl multiplied(@NotNull JvmScalarImpl b) {
        DataType ownDataType = getDataType();
        DataType scalarDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(ownDataType, scalarDataType);
        JvmTensorDataContainer dataContainer = new JvmTensorDataContainer(resultDataType, this.getShape());
        long[] shape = this.getShape();
        long nElements = ShapeUtils.getNumElements(shape);
        for (long i = 0; i < nElements; i++) {
            if (!resultDataType.isFloatingPoint()) {
                long aValue = switch (ownDataType) {
                    case INT8 -> this.getByteFlat(i);
                    case INT16 -> this.getShortFlat(i);
                    case INT32 -> this.getIntFlat(i);
                    case INT64 -> this.getLongFlat(i);
                    default -> throw new IllegalStateException("Unexpected value: " + ownDataType);
                };
                long bValue = switch (scalarDataType) {
                    case INT8 -> b.getByte();
                    case INT16 -> b.getShort();
                    case INT32 -> b.getInt();
                    case INT64 -> b.getLong();
                    default -> throw new IllegalStateException("Unexpected value: " + scalarDataType);
                };
                long cValue = aValue * bValue;
                switch (resultDataType) {
                    case INT8 -> dataContainer.setByte(i, (byte) cValue);
                    case INT16 -> dataContainer.setShort(i, (short) cValue);
                    case INT32 -> dataContainer.setInt(i, (int) cValue);
                    case INT64 -> dataContainer.setLong(i, cValue);
                    default -> throw new IllegalStateException("Unexpected value: " + resultDataType);
                }
            } else {
                double aValue = switch (ownDataType) {
                    case INT8 -> this.getByteFlat(i);
                    case INT16 -> this.getShortFlat(i);
                    case INT32 -> this.getIntFlat(i);
                    case INT64 -> this.getLongFlat(i);
                    case FLOAT32 -> this.getFloatFlat(i);
                    case FLOAT64 -> this.getDoubleFlat(i);
                };
                double bValue = switch (scalarDataType) {
                    case INT8 -> b.getByte();
                    case INT16 -> b.getShort();
                    case INT32 -> b.getInt();
                    case INT64 -> b.getLong();
                    case FLOAT32 -> b.getFloat();
                    case FLOAT64 -> b.getDouble();
                };
                double cValue = aValue * bValue;
                switch (resultDataType) {
                    case INT8 -> dataContainer.setByte(i, (byte) cValue);
                    case INT16 -> dataContainer.setShort(i, (short) cValue);
                    case INT32 -> dataContainer.setInt(i, (int) cValue);
                    case INT64 -> dataContainer.setLong(i, (long) cValue);
                    case FLOAT32 -> dataContainer.setFloat(i, (float) cValue);
                    case FLOAT64 -> dataContainer.setDouble(i, cValue);
                }
            }
        }
        return new JvmTensorImpl(dataContainer, shape);
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
                case INT8 -> this.byteData = new byte[ShapeUtils.getNumElements(shape)];
                case INT16 -> this.shortData = new short[ShapeUtils.getNumElements(shape)];
                case INT32 -> this.intData = new int[ShapeUtils.getNumElements(shape)];
                case INT64 -> this.longData = new long[ShapeUtils.getNumElements(shape)];
                case FLOAT32 -> this.floatData = new float[ShapeUtils.getNumElements(shape)];
                case FLOAT64 -> this.doubleData = new double[ShapeUtils.getNumElements(shape)];
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
                case FLOAT32 -> sb.append(String.format("%.8f", iterator.getFloat()));
                case FLOAT64 -> sb.append(String.format("%.8f", iterator.getDouble()));
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
        return new JvmTensorIterator();
    }

    @Override
    public @NotNull TensorImpl matmul(@NotNull JvmTensorImpl b) {
        long[] otherShape = b.getShape();
        if (otherShape.length != 2) {
            throw new IllegalArgumentException("matmul only supports 2D matrices");
        }
        long[] shape = getShape();
        if (shape.length != 2) {
            throw new IllegalArgumentException("matmul only supports 2D matrices");
        }
        if (shape[1] != otherShape[0]) {
            throw new IllegalArgumentException("matmul: shape mismatch. A.shape[1] != B.shape[0]");
        }
        long[] resultShape = new long[]{shape[0], otherShape[1]};
        DataType ownDataType = getDataType();
        DataType otherDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(ownDataType, otherDataType);

        JvmTensorImpl result = new JvmTensorImpl(resultDataType, resultShape);

        long[] index = new long[resultShape.length];
        for (int i = 0; i < resultShape[0]; i++) {
            for (int j = 0; j < resultShape[1]; j++) {
                if (resultDataType.isFloatingPoint()) {
                    double sum = 0;
                    for (int k = 0; k < shape[1]; k++) {
                        index[0] = i;
                        index[1] = k;
                        double aValue = switch (ownDataType) {
                            case INT8 -> getByte(index);
                            case INT16 -> getShort(index);
                            case INT32 -> getInt(index);
                            case INT64 -> getLong(index);
                            case FLOAT32 -> getFloat(index);
                            case FLOAT64 -> getDouble(index);
                        };
                        index[0] = k;
                        index[1] = j;
                        double bValue = switch (otherDataType) {
                            case INT8 -> b.getByte(index);
                            case INT16 -> b.getShort(index);
                            case INT32 -> b.getInt(index);
                            case INT64 -> b.getLong(index);
                            case FLOAT32 -> b.getFloat(index);
                            case FLOAT64 -> b.getDouble(index);
                        };
                        sum += aValue * bValue;
                    }
                    index[0] = i;
                    index[1] = j;
                    switch (resultDataType) {
                        case FLOAT32 -> result.setFloat((float) sum, index);
                        case FLOAT64 -> result.setDouble(sum, index);
                        default -> throw new IllegalStateException("Unexpected data type: " + resultDataType);
                    }
                } else {
                    long sum = 0;
                    for (int k = 0; k < shape[1]; k++) {
                        index[0] = i;
                        index[1] = k;
                        long aValue = switch (ownDataType) {
                            case INT8 -> getByte(index);
                            case INT16 -> getShort(index);
                            case INT32 -> getInt(index);
                            case INT64 -> getLong(index);
                            default -> throw new IllegalStateException("Unexpected data type: " + ownDataType);
                        };
                        index[0] = k;
                        index[1] = j;
                        long bValue = switch (otherDataType) {
                            case INT8 -> b.getByte(index);
                            case INT16 -> b.getShort(index);
                            case INT32 -> b.getInt(index);
                            case INT64 -> b.getLong(index);
                            default -> throw new IllegalStateException("Unexpected data type: " + otherDataType);
                        };
                        sum += aValue * bValue;
                    }
                    index[0] = i;
                    index[1] = j;
                    switch (resultDataType) {
                        case INT8 -> result.setByte((byte) sum, index);
                        case INT16 -> result.setShort((short) sum, index);
                        case INT32 -> result.setInt((int) sum, index);
                        case INT64 -> result.setLong(sum, index);
                        default -> throw new IllegalStateException("Unexpected data type: " + resultDataType);
                    }
                }
            }
        }
        return result;
    }

    @Override
    public @NotNull TensorImpl exp() {
        return TensorImpl.super.exp();
    }

    private class JvmTensorIterator implements ITensorIterator {

        private long flatIndex = 0;
        private final long @NotNull [] shape = getShape();
        private final long nElements = ShapeUtils.getNumElements(shape);

        @Override
        public boolean hasNext() {
            return flatIndex < nElements;
        }

        @Override
        public void moveNext() {
            flatIndex++;
        }

        @Override
        public long getNumEndingDimensions() {
            int nDims = 0;
            long flatIndex = this.flatIndex;
            for (int i = shape.length - 1; i >= 0; i--) {
                long dimSize = shape[i];
                if (flatIndex % dimSize != (dimSize - 1)) {
                    break;
                }
                flatIndex /= dimSize;
                nDims++;
            }
            return nDims;
        }

        @Override
        public long getNumStartingDimensions() {
            int nDims = 0;
            long flatIndex = this.flatIndex;
            for (int i = shape.length - 1; i >= 0; i--) {
                long dimSize = shape[i];
                if (flatIndex % dimSize != 0) {
                    break;
                }
                flatIndex /= dimSize;
                nDims++;
            }
            return nDims;
        }

        @Override
        public @NotNull DataType getDataType() {
            return JvmTensorImpl.this.getDataType();
        }

        @Override
        public byte getByte() {
            return JvmTensorImpl.this.getByteFlat(this.flatIndex);
        }

        @Override
        public short getShort() {
            return JvmTensorImpl.this.getShortFlat(this.flatIndex);
        }

        @Override
        public int getInt() {
            return JvmTensorImpl.this.getIntFlat(this.flatIndex);
        }

        @Override
        public long getLong() {
            return JvmTensorImpl.this.getLongFlat(this.flatIndex);
        }

        @Override
        public float getFloat() {
            return JvmTensorImpl.this.getFloatFlat(this.flatIndex);
        }

        @Override
        public double getDouble() {
            return JvmTensorImpl.this.getDoubleFlat(this.flatIndex);
        }
    }
}
