package me.mikex86.scicore;

import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

public class DefaultTensorIterator implements ITensorIterator {

    private final ITensor tensor;
    private long flatIndex = 0;
    private final long @NotNull [] shape;
    private final long nElements;

    public DefaultTensorIterator(ITensor tensor) {
        this.tensor = tensor;
        this.shape = tensor.getShape();
        this.nElements = ShapeUtils.getNumElements(shape);
    }

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
        return tensor.getDataType();
    }

    @Override
    public byte getByte() {
        return tensor.getByteFlat(this.flatIndex);
    }

    @Override
    public short getShort() {
        return tensor.getShortFlat(this.flatIndex);
    }

    @Override
    public int getInt() {
        return tensor.getIntFlat(this.flatIndex);
    }

    @Override
    public long getLong() {
        return tensor.getLongFlat(this.flatIndex);
    }

    @Override
    public float getFloat() {
        return tensor.getFloatFlat(this.flatIndex);
    }

    @Override
    public double getDouble() {
        return tensor.getDoubleFlat(this.flatIndex);
    }
}
