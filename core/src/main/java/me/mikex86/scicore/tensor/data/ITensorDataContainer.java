package me.mikex86.scicore.tensor.data;

import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.utils.dispose.IDisposable;
import org.jetbrains.annotations.NotNull;

import java.nio.*;

public interface ITensorDataContainer extends IDisposable {
    byte getInt8Flat(long flatIndex);

    void setInt8Flat(long flatIndex, byte value);

    short getInt16Flat(long flatIndex);

    void setInt16Flat(long flatIndex, short value);

    int getInt32Flat(long flatIndex);

    void setInt32Flat(long flatIndex, int value);

    long getInt64Flat(long flatIndex);

    void setInt64Flat(long flatIndex, long value);

    float getFloat32Flat(long flatIndex);

    void setFloat32Flat(long flatIndex, float value);

    double getFloat64Flat(long flatIndex);

    void setFloat64Flat(long flatIndex, double value);

    void setBooleanFlat(long flatIndex, boolean value);

    boolean getBooleanFlat(long flatIndex);

    void setContents(long startIndex, @NotNull ByteBuffer buffer);

    default void setContents(@NotNull ByteBuffer buffer) {
        setContents(0, buffer);
    }

    void setContents(long startIndex, @NotNull ShortBuffer buffer);

    default void setContents(@NotNull ShortBuffer buffer) {
        setContents(0, buffer);
    }

    void setContents(long startIndex, @NotNull IntBuffer buffer);

    default void setContents(@NotNull IntBuffer buffer) {
        setContents(0, buffer);
    }

    void setContents(long startIndex, @NotNull LongBuffer buffer);

    default void setContents(@NotNull LongBuffer buffer) {
        setContents(0, buffer);
    }

    void setContents(long startIndex, @NotNull FloatBuffer buffer);

    default void setContents(@NotNull FloatBuffer buffer) {
        setContents(0, buffer);
    }

    void setContents(long startIndex, @NotNull DoubleBuffer buffer);

    default void setContents(@NotNull DoubleBuffer buffer) {
        setContents(0, buffer);
    }

    void setContents(long flatIndex, boolean @NotNull [] data);

    default void setContents(boolean @NotNull [] data) {
        setContents(0, data);
    }

    void fillRegion(long startFlatIndex, long endFlatIndex, byte value);

    default void fill(byte value) {
        fillRegion(0, getNumberOfElements(), value);
    }

    void fillRegion(long startFlatIndex, long endFlatIndex, short value);

    default void fill(short value) {
        fillRegion(0, getNumberOfElements(), value);
    }

    void fillRegion(long startFlatIndex, long endFlatIndex, int value);

    default void fill(int value) {
        fillRegion(0, getNumberOfElements(), value);
    }

    void fillRegion(long startFlatIndex, long endFlatIndex, long value);

    default void fill(long value) {
        fillRegion(0, getNumberOfElements(), value);
    }

    void fillRegion(long startFlatIndex, long endFlatIndex, float value);

    default void fill(float value) {
        fillRegion(0, getNumberOfElements(), value);
    }

    void fillRegion(long startFlatIndex, long endFlatIndex, double value);

    default void fill(double value) {
        fillRegion(0, getNumberOfElements(), value);
    }

    void fillRegion(long startFlatIndex, long endFlatIndex, boolean value);

    default void fill(boolean value) {
        fillRegion(0, getNumberOfElements(), value);
    }


    /**
     * Returns a reference memory handle to the internal data buffer in the specified index range.
     *
     * @param startFlatIndex the start index of the data to copy (flat index)
     * @param endFlatIndex   the end index of the data to copy (exclusive, flat index)
     * @return the host byte buffer. Must not be freed.
     */
    @NotNull DirectMemoryHandle getAsDirectBuffer(long startFlatIndex, long endFlatIndex);

    /**
     * @return a reference memory handle to the internal data buffer, when possible.
     */
    @NotNull DirectMemoryHandle getAsDirectBuffer();

    @NotNull DataType getDataType();

    /**
     * @return the size of the data container in bytes
     */
    long getDataSize();

    long getNumberOfElements();

    /**
     * Increments the reference count of the data container.
     */
    void incRc();

    /**
     * Decrements the reference count of the data container.
     */
    void decRc();
}
