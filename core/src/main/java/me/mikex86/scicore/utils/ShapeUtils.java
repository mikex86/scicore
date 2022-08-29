package me.mikex86.scicore.utils;

import org.jetbrains.annotations.NotNull;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Utility functions operating on shapes.
 * <p>
 * Shapes are represented by a long array.
 * shape[0] is the size of the top most dimension (highest dimension),
 * shape[shape.length - 1] is the size of the lowest dimension (the level of the scalar).
 */
public class ShapeUtils {

    /**
     * Computes the number of elements in a shape.
     *
     * @param shape the specified shape
     * @return the number of elements in the shape
     */
    public static long getNumElements(long @NotNull [] shape) {
        long numElements = 1;
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
     * Strides are defined as the number of scalars
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
     * @param index   the n-dimensional index. indices.length may be less than strides.length, when the offset
     *                to the first element of a particular dimension that is not the lowest-level dimension (individual scalars).
     * @param strides the strides as defined byte {@link #makeStrides(long[])}
     * @return the flat index
     */
    public static long getFlatIndex(long @NotNull [] index, long @NotNull [] strides) {
        if (index.length > strides.length) {
            throw new IllegalArgumentException("Indices length must be less than or equal to strides length");
        }
        long flatIndex = 0;
        for (int dim = 0; dim < index.length; dim++) {
            flatIndex += index[dim] * strides[dim];
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

    /**
     * Increments an n-dimensional index constrained by a specific shape.
     *
     * @param shape the shape to constrain the index to
     * @param index the index to increment (will be modified in place)
     * @return the true if the index was incremented, false if the index was already at the last element of the shape
     */
    public static boolean incrementIndex(long @NotNull [] shape, long @NotNull [] index) {
        for (int dim = index.length - 1; dim >= 0; dim--) {
            if (index[dim] < shape[dim] - 1) {
                index[dim]++;
                return true;
            }
            index[dim] = 0;
        }
        return false;
    }

    /**
     * Returns the larger of the two supplied shapes.
     *
     * @param shapeA the first shape
     * @param shapeB the second shape
     * @return the larger shape
     * @throws IllegalArgumentException if the two shapes are not broadcast-able
     * @see <a href="https://numpy.org/doc/stable/user/basics.broadcasting.html">Numpy Broadcasting</a>
     */
    public static long @NotNull [] broadcastShapes(long @NotNull [] shapeA, long @NotNull [] shapeB) {
        long[] broadcastShape;

        if (shapeB.length > shapeA.length) {
            // swap shapes, make shapeA always larger
            // this is just a reference swap because java, making this cheap
            long[] tmp = shapeB;
            shapeB = shapeA;
            shapeA = tmp;
        }

        broadcastShape = new long[shapeA.length];
        for (int i = 0; i < shapeA.length; i++) {
            long elementA = shapeA[shapeA.length - 1 - i];
            long elementB = i < shapeB.length ? shapeB[shapeB.length - 1 - i] : -1;
            long dimSize;
            if (elementA == elementB || elementB == 1 || elementB == -1) {
                dimSize = elementA;
            } else if (elementA == 1) {
                dimSize = elementB;
            } else {
                throw new IllegalArgumentException("Shapes are not broadcast-able: shapeA: " + ShapeUtils.toString(shapeA) + ", shapeB: " + ShapeUtils.toString(shapeB));
            }
            broadcastShape[broadcastShape.length - 1 - i] = dimSize;
        }

        return broadcastShape;
    }

    /**
     * Constrains the index in the specified shape. This means that every dimension of the index will be modulo-ed by the
     * length of said dimension as specified at shape[dimension].
     *
     * @param index the index to constrain. The output will be written back into this array.
     * @param shape the shape to constrain the index in.
     */
    public static void constrainIndex(long @NotNull [] index, long @NotNull [] shape) {
        if (index.length != shape.length) {
            throw new IllegalArgumentException("index.length (" + index.length + ") must match shape.length (" + shape.length + ")");
        }
        for (int i = 0; i < shape.length; i++) {
            index[i] = index[i] % shape[i];
        }
    }

    /**
     * Returns the shape of the supplied java array.
     *
     * @param array the supplied java array. The shape of the array must conform to a tensor (all elements of given dimension must have the same size).
     * @return the shape of the java array
     */
    public static long @NotNull [] getArrayShape(@NotNull Object array) {
        List<Long> shape = new ArrayList<>();
        Object current = array;
        while (current.getClass().isArray()) {
            int currentLength = Array.getLength(current);
            shape.add((long) currentLength);

            Object firstOfDim = Array.get(current, 0);
            if (firstOfDim.getClass().isArray()) {
                int firstOfDimLength = Array.getLength(firstOfDim);
                // iterate over current dimension and check if all elements are arrays of the same length (check if match first element)
                for (int i = 1; i < currentLength; i++) {
                    Object otherArrayOnSameDim = Array.get(current, i);
                    int lengthOfOtherArrayOnSameDim = Array.getLength(otherArrayOnSameDim);
                    if (lengthOfOtherArrayOnSameDim != firstOfDimLength) {
                        throw new IllegalArgumentException("Array must be n-dimensional");
                    }
                }
            }
            current = firstOfDim;
        }
        long[] shapeArray = new long[shape.size()];
        for (int i = 0; i < shape.size(); i++) {
            shapeArray[i] = shape.get(i);
        }
        return shapeArray;
    }

    public static boolean isScalar(long @NotNull [] shape) {
        return shape.length == 0 || ShapeUtils.getNumElements(shape) == 1;
    }

    public static long @NotNull [] getCommonShape(long @NotNull [] shapeA, long @NotNull [] shapeB) {
        int commonLength = Math.min(shapeA.length, shapeB.length);
        for (int i = 0; i < commonLength; i++) {
            if (shapeA[shapeA.length - 1 - i] != shapeB[shapeB.length - 1 - i]) {
                return Arrays.copyOfRange(shapeA, shapeA.length - i, shapeA.length);
            }
        }
        return Arrays.copyOfRange(shapeA, 0, commonLength);
    }

    public static int getNumCommonDimensions(long @NotNull [] shapeA, long @NotNull [] shapeB) {
        int commonLength = Math.min(shapeA.length, shapeB.length);
        for (int i = 0; i < commonLength; i++) {
            if (shapeA[shapeA.length - 1 - i] != shapeB[shapeB.length - 1 - i]) {
                return i;
            }
        }
        return commonLength;
    }

    public static int getNumNotCommonDimensions(long @NotNull [] shapeA, long @NotNull [] shapeB) {
        return Math.max(shapeA.length, shapeB.length) - getNumCommonDimensions(shapeA, shapeB);
    }

    /**
     * Returns the n-th higher dimension index than the supplied index in the context of the supplied shape.
     * Higher indices have lower numeric value. For example the dimension index=0 is the highest dimension of the shape,
     * while index=shape.length-1 is the lowest dimension of the shape.
     * Note that for index=0 the n-th higher dimension is {@code shape.length} - n. This mimics python {@code array[-1]} indexing referencing the last element.
     * This behavior is there because it is useful.
     * @param dimension the current dimension index
     * @param nDimensions the number of dimensions in the shape ({@code shape.length})
     * @param n the number of dimensions the dimension index should get higher by
     * @return the new dimension index
     */
    public static int getHigherDimension(int dimension, int nDimensions, int n) {
        int newDimension = dimension - (n % nDimensions);
        if (newDimension < 0) {
            newDimension += nDimensions;
        }
        return newDimension;
    }

    /**
     * Compares two shapes and returns whether 'shapeA broadcasts to shapeB' or vice versa, meaning whether shapeA is a subset of shapeB in the sense of a broadcast or not.
     * @see #broadcastShapes(long[], long[]) for more information on broadcasting.
     * @param shapeA the first shape to compare
     * @param shapeB the second shape to compare
     * @return -1 if shapeA < shapeB, 0 if shapeA == shapeB, 1 if shapeA > shapeB, where > means superset, < means subset, and == means the shapes are equivalent.
     * @throws IllegalArgumentException if the shapes are not broadcast-able
     */
    public static int compareBroadcastRank(long@NotNull[] shapeA, long@NotNull[] shapeB) {
        int maxLength = Math.max(shapeA.length, shapeB.length);
        for (int i = 0; i < maxLength; i++) {
            long elementA = i < shapeA.length ? shapeA[shapeA.length - 1 - i] : -1;
            long elementB = i < shapeB.length ? shapeB[shapeB.length - 1 - i] : -1;
            if (elementA != elementB) {
                if (elementA == -1) {
                    return -1; // shapeA is a subset of shapeB
                } else if (elementB == -1) {
                    return 1; // shapeB is a subset of shapeA
                } else if (elementA == 1) {
                    return -1; // shapeA is a subset of shapeB
                } else if (elementB == 1) {
                    return 1; // shapeB is a subset of shapeA
                } else {
                    throw new IllegalArgumentException("Shapes are not broadcast-able: shapeA: " + ShapeUtils.toString(shapeA) + ", shapeB: " + ShapeUtils.toString(shapeB));
                }
            }
        }
        return 0;
    }
}