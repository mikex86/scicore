package me.mikex86.scicore.tensor.data;

import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.utils.dispose.IDisposable;
import org.jetbrains.annotations.NotNull;

import java.nio.*;

public interface ITensorDataContainer extends IDisposable {
    byte getInt8Flat(long flatIndex);

    void getInt8Flat(byte value, long flatIndex);

    short getInt16Flat(long flatIndex);

    void setInt16Flat(short value, long flatIndex);

    int getInt32Flat(long flatIndex);

    void setInt32Flat(int value, long flatIndex);

    long getInt64Flat(long flatIndex);

    void setInt64Flat(long value, long flatIndex);

    float getFloat32Flat(long flatIndex);

    void setFloat32Flat(float value, long flatIndex);

    double getFloat64Flat(long flatIndex);

    void setFloat64Flat(double value, long flatIndex);

    void setBooleanFlat(boolean value, long flatIndex);

    boolean getBooleanFlat(long flatIndex);

    void setContents(@NotNull ByteBuffer buffer);

    void setContents(@NotNull ShortBuffer buffer);

    void setContents(@NotNull IntBuffer buffer);

    void setContents(@NotNull LongBuffer buffer);

    void setContents(@NotNull FloatBuffer buffer);

    void setContents(@NotNull DoubleBuffer buffer);

    void setContents(boolean @NotNull [] data);

    void fill(byte value);

    void fill(short value);

    void fill(int value);

    void fill(long value);

    void fill(float value);

    void fill(double value);

    void fill(boolean value);

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
}
