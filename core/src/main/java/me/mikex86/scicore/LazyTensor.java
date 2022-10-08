package me.mikex86.scicore;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.op.IDerivedTensor;
import me.mikex86.scicore.utils.Pair;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.nio.*;
import java.util.Arrays;
import java.util.function.Supplier;

/**
 * Encapsulates tensor supplier. Lazily evaluates the supplier when the result is needed.
 */
public class LazyTensor extends AbstractTensor implements IDerivedTensor {

    private final long @NotNull [] resultShape;

    @NotNull
    private final DataType resultDataType;

    @NotNull
    private final Supplier<ITensor> resultSupplier;

    @Nullable
    private ITensor lazyResult;

    @NotNull
    private final ISciCoreBackend sciCoreBackend;

    public LazyTensor(@NotNull ISciCoreBackend backend, long @NotNull [] resultShape, @NotNull DataType resultDataType, @NotNull Supplier<ITensor> resultSupplier) {
        this.numElements = ShapeUtils.getNumElements(resultShape);
        this.resultShape = resultShape;
        this.resultDataType = resultDataType;
        this.resultSupplier = resultSupplier;
        this.lazyResult = resultSupplier.get();
        this.sciCoreBackend = backend;
    }

    @NotNull
    @Override
    public ITensor result() {
        if (lazyResult == null) {
            lazyResult = resultSupplier.get();
        }
        return lazyResult;
    }

    @Override
    public @NotNull DataType getDataType() {
        return resultDataType;
    }

    @Override
    public long @NotNull [] getShape() {
        return resultShape;
    }

    @Override
    public long @NotNull [] getStrides() {
        return result().getStrides();
    }

    @Override
    public short getShort(long @NotNull ... indices) {
        return result().getShort(indices);
    }

    @Override
    public int getInt(long @NotNull ... indices) {
        return result().getInt(indices);
    }

    @Override
    public long getLong(long @NotNull ... indices) {
        return result().getLong(indices);
    }

    @Override
    public boolean getBooleanFlat(long flatIndex) {
        return result().getBooleanFlat(flatIndex);
    }

    @Override
    public void setBooleanFlat(boolean value, long flatIndex) {
        result().setBooleanFlat(value, flatIndex);
    }

    @Override
    public byte getByteFlat(long flatIndex) {
        return result().getByteFlat(flatIndex);
    }

    @Override
    public short getShortFlat(long flatIndex) {
        return result().getShortFlat(flatIndex);
    }

    @Override
    public int getIntFlat(long flatIndex) {
        return result().getIntFlat(flatIndex);
    }

    @Override
    public long getLongFlat(long flatIndex) {
        return result().getLongFlat(flatIndex);
    }

    @Override
    public float getFloatFlat(long flatIndex) {
        return result().getFloatFlat(flatIndex);
    }

    @Override
    public double getDoubleFlat(long flatIndex) {
        return result().getDoubleFlat(flatIndex);
    }

    @Override
    public void setByteFlat(byte value, long flatIndex) {
        result().setByteFlat(value, flatIndex);
    }

    @Override
    public void setShortFlat(short value, long flatIndex) {
        result().setShortFlat(value, flatIndex);
    }

    @Override
    public void setIntFlat(int value, long flatIndex) {
        result().setIntFlat(value, flatIndex);
    }

    @Override
    public void setLongFlat(long value, long flatIndex) {
        result().setLongFlat(value, flatIndex);
    }

    @Override
    public void setFloatFlat(float value, long flatIndex) {
        result().setFloatFlat(value, flatIndex);
    }

    @Override
    public void setDoubleFlat(double value, long flatIndex) {
        result().setDoubleFlat(value, flatIndex);
    }

    @Override
    public @NotNull ITensor copy() {
        return result().copy();
    }

    @Override
    public void setContents(@NotNull ITensor tensor) {
        result().setContents(tensor);
    }

    @Override
    public void setContents(@NotNull ByteBuffer buffer) {
        result().setContents(buffer);
    }

    @Override
    public void setContents(@NotNull ShortBuffer buffer) {
        result().setContents(buffer);
    }

    @Override
    public void setContents(@NotNull IntBuffer buffer) {
        result().setContents(buffer);
    }

    @Override
    public void setContents(@NotNull LongBuffer buffer) {
        result().setContents(buffer);
    }

    @Override
    public void setContents(@NotNull FloatBuffer buffer) {
        result().setContents(buffer);
    }

    @Override
    public void setContents(@NotNull DoubleBuffer buffer) {
        result().setContents(buffer);
    }

    @Override
    public void setContents(boolean @NotNull [] buffer) {
        result().setContents(buffer);
    }

    @Override
    public void setContents(long @NotNull [] index, @NotNull ITensor tensor, boolean useView) {
        result().setContents(index, tensor, useView);
    }

    @Override
    public void fill(byte i) {
        result().fill(i);
    }

    @Override
    public void fill(short i) {
        result().fill(i);
    }

    @Override
    public void fill(int i) {
        result().fill(i);
    }

    @Override
    public void fill(long i) {
        result().fill(i);
    }

    @Override
    public void fill(float i) {
        result().fill(i);
    }

    @Override
    public void fill(double i) {
        result().fill(i);
    }

    @Override
    public void fill(boolean value) {
        result().fill(value);
    }

    @Override
    public @NotNull ISciCoreBackend getSciCoreBackend() {
        return this.sciCoreBackend;
    }

    @Override
    public @NotNull DirectMemoryHandle getContentsAsDirectMemory() {
        return result().getContentsAsDirectMemory();
    }

    @Override
    public @NotNull DirectMemoryHandle getContentsAsDirectMemory(long startFlatIndex, long endFlatIndex) {
        return result().getContentsAsDirectMemory(startFlatIndex, endFlatIndex);
    }

    @Override
    public String toString() {
        return result().toString();
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof ITensor tensor)) {
            return false;
        }
        if (!Arrays.equals(tensor.getShape(), this.getShape())) {
            return false;
        }
        if (tensor.getDataType() != this.getDataType()) {
            return false;
        }
        if (obj == this) {
            return true;
        }
        return result().equals(obj);
    }

    @Override
    public <T extends ITensor> @Nullable T getIfIsType(@NotNull Class<T> typeClass) {
        return result().getIfIsType(typeClass);
    }
}
