package me.mikex86.scicore;

import org.jetbrains.annotations.NotNull;

/**
 * An iterator iterating through the values of a tensor.
 * The order of iteration arises by incrementing a flat index rather than an n-dimensional index.
 * This flat index ranges from 0 to the number of elements in the tensor, where the number of elements is the product of the dimensions.
 */
public interface ITensorIterator {

    boolean hasNext();

    void moveNext();

    /**
     * @return the number of dimensions that the current element is the last element of, while
     * the value only counts, if there is no lower dimension that is still in range.
     * For example, if the tensor is (2, 3, 4) and the current index is (1, 2, 3), the value returned is 2,
     * while if the tensor is (2, 3, 4) and the current index is (1, 2, 2), the value returned is not 1 but 0, because
     */
    long getNumEndingDimensions();

    long getNumStartingDimensions();

    @NotNull
    DataType getDataType();

    byte getByte();

    short getShort();

    int getInt();

    long getLong();

    float getFloat();

    double getDouble();

    boolean getBoolean();
}
