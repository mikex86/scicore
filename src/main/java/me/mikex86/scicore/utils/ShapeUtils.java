package me.mikex86.scicore.utils;

import org.jetbrains.annotations.NotNull;

import java.util.Arrays;

/**
 * Utility functions operating on shapes.
 * <p>
 * Shapes are represented by a long array.
 * shape[0] is the size of the top most dimension,
 * shape[shape.length - 1] is the size of the lowest dimension (the level of the scalar).
 */
public class ShapeUtils {

    /**
     * Computes the number of elements in a shape.
     *
     * @param shape the specified shape
     * @return the number of elements in the shape
     */
    public static int getNumElements(long @NotNull [] shape) {
        int numElements = 1;
        for (long l : shape) {
            numElements *= l;
        }
        return numElements;
    }

    /**
     * Computes the number of elements in a shape up to a specified dimension.
     *
     * @param shape the specified shape
     * @param dim   the specified dimension
     * @return the number of elements in the shape
     */
    public static int getNumElements(long @NotNull [] shape, int dim) {
        int numElements = 1;
        for (int i = 0; i < shape.length; i++) {
            if (i > dim) {
                break;
            }
            long l = shape[i];
            numElements *= l;
        }
        return numElements;
    }

    /**
     * Computes the strides for a specified shape.
     * Strides ar the defined as the number of scalars
     * (value in the lowest dimension (the dimension with size shape[shape.length -1]))
     * to jump from one element to the next in a specified dimension.
     * This is used to compute the flat index of an element in an
     * n-dimensional array such that the flat index is the sum of
     * the strides of the dimensions multiplied by the index at the specified dimension.
     *
     * @param shape the specified shape
     * @return the strides for the specified shape
     */
    public static long @NotNull [] makeStrides(long @NotNull [] shape) {
        long[] strides = new long[shape.length];
        for (int dim = shape.length - 1; dim >= 0; dim--) {
            if (dim == shape.length - 1) {
                strides[dim] = 1;
            } else {
                strides[dim] = strides[dim + 1] * shape[dim + 1];
            }
        }
        return strides;
    }

    /**
     * Computes a flat index from a n-dimension index given an array of strides.
     *
     * @param indices the n-dimensional index. indices.length may be less than strides.length, when the offset
     *                to the first element of a particular dimension that is not the lowest-level dimension (individual scalars).
     * @param strides the strides as defined byte {@link #makeStrides(long[])}
     * @return the flat index
     */
    public static long getFlatIndex(long @NotNull [] indices, long @NotNull [] strides) {
        if (indices.length > strides.length) {
            throw new IllegalArgumentException("Indices length must be less than or equal to strides length");
        }
        long flatIndex = 0;
        for (int dim = 0; dim < indices.length; dim++) {
            flatIndex += indices[dim] * strides[dim];
        }
        return flatIndex;
    }

    /**
     * Creates a string representation of a shape.
     *
     * @param shape the specified shape
     * @return the string representation of the shape
     */
    @NotNull
    public static String toString(long @NotNull [] shape) {
        StringBuilder sb = new StringBuilder();
        sb.append("(");
        for (int dim = 0; dim < shape.length; dim++) {
            sb.append(shape[dim]);
            if (dim < shape.length - 1) {
                sb.append(", ");
            }
        }
        sb.append(")");
        return sb.toString();
    }

    /**
     * @param shapeA the first shape
     * @param shapeB the second shape
     * @return true if the two shapes are equal
     */
    public static boolean equals(long @NotNull [] shapeA, long @NotNull [] shapeB) {
        return Arrays.equals(shapeA, shapeB);
    }
}
