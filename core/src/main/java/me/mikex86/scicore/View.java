package me.mikex86.scicore;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.nio.*;
import java.util.Arrays;

class View extends AbstractTensor {

    private final @NotNull ITensor viewed;
    private final long @NotNull [] shape;
    private final long offset;
    private final long[] localStrides;

    View(@NotNull ITensor viewed, long @NotNull [] shape, long offset, long[] localStrides) {
        this.viewed = viewed;
        this.shape = shape;
        this.offset = offset;
        this.localStrides = localStrides;
    }

    @Override
    public @NotNull DataType getDataType() {
        return this.viewed.getDataType();
    }

    @Override
    public long @NotNull [] getShape() {
        return this.shape;
    }

    @Override
    public long @NotNull [] getStrides() {
        return this.localStrides;
    }

    @Override
    public byte getByte(long @NotNull ... indices) {
        validateDataType(DataType.INT8);
        validateIndices(indices);
        long flatIndex = ShapeUtils.getFlatIndex(indices, this.localStrides);
        return this.viewed.getByteFlat(this.offset + flatIndex);
    }

    @Override
    public short getShort(long @NotNull ... indices) {
        validateDataType(DataType.INT16);
        validateIndices(indices);
        long flatIndex = ShapeUtils.getFlatIndex(indices, this.localStrides);
        return this.viewed.getShortFlat(this.offset + flatIndex);
    }

    @Override
    public int getInt(long @NotNull ... indices) {
        validateDataType(DataType.INT32);
        validateIndices(indices);
        long flatIndex = ShapeUtils.getFlatIndex(indices, this.localStrides);
        return this.viewed.getIntFlat(this.offset + flatIndex);
    }

    @Override
    public long getLong(long @NotNull ... indices) {
        validateDataType(DataType.INT64);
        validateIndices(indices);
        long flatIndex = ShapeUtils.getFlatIndex(indices, this.localStrides);
        return this.viewed.getLongFlat(this.offset + flatIndex);
    }

    @Override
    public float getFloat(long @NotNull ... indices) {
        validateDataType(DataType.FLOAT32);
        validateIndices(indices);
        long flatIndex = ShapeUtils.getFlatIndex(indices, this.localStrides);
        return this.viewed.getFloatFlat(this.offset + flatIndex);
    }

    @Override
    public double getDouble(long @NotNull ... indices) {
        validateDataType(DataType.FLOAT64);
        validateIndices(indices);
        long flatIndex = ShapeUtils.getFlatIndex(indices, this.localStrides);
        return this.viewed.getDoubleFlat(this.offset + flatIndex);
    }

    @Override
    public void setByte(byte value, long @NotNull ... indices) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setShort(short value, long @NotNull ... indices) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setInt(int value, long @NotNull ... indices) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setLong(long value, long @NotNull ... indices) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setFloat(float value, long @NotNull ... indices) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setDouble(double value, long @NotNull ... indices) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public boolean getBooleanFlat(long flatIndex) {
        validateDataType(DataType.BOOLEAN);
        return this.viewed.getBooleanFlat(this.offset + flatIndex);
    }

    @Override
    public void setBooleanFlat(boolean value, long flatIndex) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public boolean getBoolean(long @NotNull ... indices) {
        validateDataType(DataType.BOOLEAN);
        validateIndices(indices);
        long flatIndex = ShapeUtils.getFlatIndex(indices, this.localStrides);
        return this.viewed.getBooleanFlat(this.offset + flatIndex);
    }

    @Override
    public void setBoolean(boolean value, long @NotNull ... indices) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public byte getByteFlat(long flatIndex) {
        return this.viewed.getByteFlat(this.offset + flatIndex);
    }

    @Override
    public short getShortFlat(long flatIndex) {
        return this.viewed.getShortFlat(this.offset + flatIndex);
    }

    @Override
    public int getIntFlat(long flatIndex) {
        return this.viewed.getIntFlat(this.offset + flatIndex);
    }

    @Override
    public long getLongFlat(long flatIndex) {
        return this.viewed.getLongFlat(this.offset + flatIndex);
    }

    @Override
    public float getFloatFlat(long flatIndex) {
        return this.viewed.getFloatFlat(this.offset + flatIndex);
    }

    @Override
    public double getDoubleFlat(long flatIndex) {
        return this.viewed.getDoubleFlat(this.offset + flatIndex);
    }

    @Override
    public void setByteFlat(byte value, long flatIndex) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setShortFlat(short value, long flatIndex) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setIntFlat(int value, long flatIndex) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setLongFlat(long value, long flatIndex) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setFloatFlat(float value, long flatIndex) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setDoubleFlat(double value, long flatIndex) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public @NotNull ITensor copy() {
        return this;
    }

    @Override
    public void setContents(@NotNull ITensor tensor) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setContents(@NotNull ByteBuffer buffer) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setContents(@NotNull ShortBuffer buffer) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setContents(@NotNull IntBuffer buffer) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setContents(@NotNull LongBuffer buffer) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setContents(@NotNull FloatBuffer buffer) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setContents(@NotNull DoubleBuffer buffer) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setContents(boolean @NotNull [] buffer) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void setContents(long @NotNull [] index, @NotNull ITensor tensor, boolean useView) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void fill(byte i) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void fill(short i) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void fill(int i) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void fill(long i) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void fill(float i) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void fill(double i) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public void fill(boolean value) {
        throw new UnsupportedOperationException("Views are read-only");
    }

    @Override
    public @NotNull ITensorIterator iterator() {
        return new DefaultTensorIterator(this);
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof ITensor other)) {
            return false;
        }
        long[] shape = getShape();
        if (!ShapeUtils.equals(shape, other.getShape())) {
            return false;
        }
        long nElements = ShapeUtils.getNumElements(shape);
        boolean oneIsFloatingPoint = getDataType().isFloatingPoint() || other.getDataType().isFloatingPoint();
        if (oneIsFloatingPoint) {
            for (long i = 0; i < nElements; i++) {
                double a = getAsDoubleFlat(i);
                double b = other.getAsDoubleFlat(i);
                if (Math.abs(a - b) > EPSILON) {
                    return false;
                }
            }
        } else {
            for (long i = 0; i < nElements; i++) {
                long a = getAsLongFlat(i);
                long b = getAsLongFlat(i);
                if (a != b) {
                    return false;
                }
            }
        }
        return true;
    }


    @Override
    public String toString() {
        long[] shape = getShape();
        StringBuilder sb = new StringBuilder("View(dtype=" + getDataType() + ", shape=" + Arrays.toString(shape) + ", data=\n");
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
    public @NotNull ISciCoreBackend getSciCoreBackend() {
        return this.viewed.getSciCoreBackend();
    }
}
