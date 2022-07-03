package me.mikex86.scicore.utils;

import org.jetbrains.annotations.NotNull;

import java.lang.reflect.Array;
import java.util.ArrayDeque;
import java.util.Deque;

public class ArrayUtils {

    /**
     * Returns the elements of the array flattened into a single dimension.
     *
     * @param array the array to flatten. The array must conform to a tensor (all elements of given dimension must have the same size).
     * @return the flattened array
     */
    public static Object getElementsFlat(Object array) {
        long[] shape = ShapeUtils.getArrayShape(array);
        long nElements = ShapeUtils.getNumElements(shape);
        if (nElements >= Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Array is too large to flatten");
        }
        Object elements = null;
        Deque<Object> stack = new ArrayDeque<>(Math.toIntExact(nElements));
        stack.push(array);
        int index = 0;
        while (!stack.isEmpty()) {
            Object current = stack.pollLast();
            int length = Array.getLength(current);
            if (length > 0) {
                if (current instanceof byte[] currentArray) {
                    if (elements == null) {
                        elements = new byte[(int) nElements];
                    } else if (!(elements instanceof byte[])) {
                        throw new IllegalArgumentException("Mixed data types in array");
                    }
                    System.arraycopy(currentArray, 0, (byte[]) elements, index, length);
                    index += length;
                } else if (current instanceof short[] currentArray) {
                    if (elements == null) {
                        elements = new short[(int) nElements];
                    } else if (!(elements instanceof short[])) {
                        throw new IllegalArgumentException("Mixed data types in array");
                    }
                    System.arraycopy(currentArray, 0, (short[]) elements, index, length);
                    index += length;
                } else if (current instanceof int[] currentArray) {
                    if (elements == null) {
                        elements = new int[(int) nElements];
                    } else if (!(elements instanceof int[])) {
                        throw new IllegalArgumentException("Mixed data types in array");
                    }
                    System.arraycopy(currentArray, 0, (int[]) elements, index, length);
                    index += length;
                } else if (current instanceof long[] currentArray) {
                    if (elements == null) {
                        elements = new long[(int) nElements];
                    } else if (!(elements instanceof long[])) {
                        throw new IllegalArgumentException("Mixed data types in array");
                    }
                    System.arraycopy(currentArray, 0, (long[]) elements, index, length);
                    index += length;
                } else if (current instanceof float[] currentArray) {
                    if (elements == null) {
                        elements = new float[(int) nElements];
                    } else if (!(elements instanceof float[])) {
                        throw new IllegalArgumentException("Mixed data types in array");
                    }
                    System.arraycopy(currentArray, 0, (float[]) elements, index, length);
                    index += length;
                } else if (current instanceof double[] currentArray) {
                    if (elements == null) {
                        elements = new double[(int) nElements];
                    } else if (!(elements instanceof double[])) {
                        throw new IllegalArgumentException("Mixed data types in array");
                    }
                    System.arraycopy(currentArray, 0, (double[]) elements, index, length);
                    index += length;
                } else {
                    for (int j = 0; j < length; j++) {
                        Object element = Array.get(current, j);
                        if (element.getClass().isArray()) {
                            stack.push(element);
                        }
                    }
                }
            }
        }
        return elements;
    }

    /**
     * Returns the component type of the array. Can be n-dimensional.
     *
     * @param array the array to get the component type of.
     * @return the component type of the array.
     */
    @NotNull
    public static Class<?> getComponentType(@NotNull Object array) {
        Class<?> componentType = array.getClass();
        while (componentType.isArray()) {
            componentType = componentType.getComponentType();
        }
        return componentType;
    }

    /**
     * Returns true if arrayA contains arrayB.
     * This means that arrayB is a subset of arrayA.
     *
     * @param arrayA the array to check if it contains arrayB.
     * @param arrayB the array to check if it is a subset of arrayA.
     * @return true if arrayA contains arrayB.
     */
    public static boolean contains(long @NotNull [] arrayA, long @NotNull [] arrayB) {
        if (arrayA.length < arrayB.length) {
            return false;
        }
        int i = 0;
        int j = 0;
        // search for the first element of arrayB in arrayA
        while (j < arrayA.length && arrayB[i] != arrayA[j]) {
            j++;
        }
        // check if n elements of arrayB follow the first element of arrayA
        for (i = 0; i < arrayB.length; i++) {
            if (arrayB[i] != arrayA[j + i]) {
                return false;
            }
        }
        return true;
    }


    /**
     * Checks if the last elements of arrayA match that of arrayB.
     * @param arrayA the array to check if it ends with arrayB.
     * @param arrayB the subset that arrayA should end with.
     * @return true if arrayA ends with arrayB.
     */
    public static boolean endsWidth(long @NotNull [] arrayA, long @NotNull [] arrayB) {
        if (arrayA.length < arrayB.length) {
            return false;
        }
        for (int i = 0; i < arrayB.length; i++) {
            if (arrayA[arrayA.length - 1 - i] != arrayB[arrayB.length - 1 - i]) {
                return false;
            }
        }
        return true;
    }
}
