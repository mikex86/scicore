package me.mikex86.scicore.utils;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ArrayUtilsTest {

    @Test
    void getElementsFlat_byte1d() {
        byte[] array = {1, 2, 3, 4};
        byte[] expected = {1, 2, 3, 4};
        byte[] actual = (byte[]) ArrayUtils.getElementsFlat(array);
        assertArrayEquals(expected, actual);
    }

    @Test
    void getElementsFlat_byte2d() {
        byte[][] array = {{1, 2, 3, 4}, {5, 6, 7, 8}};
        byte[] expected = {1, 2, 3, 4, 5, 6, 7, 8};
        byte[] actual = (byte[]) ArrayUtils.getElementsFlat(array);
        assertArrayEquals(expected, actual);
    }

    @Test
    void getElementsFlat_byte3d() {
        byte[][][] array = {{{1, 2, 3, 4}, {5, 6, 7, 8}}, {{9, 10, 11, 12}, {13, 14, 15, 16}}};
        byte[] expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        byte[] actual = (byte[]) ArrayUtils.getElementsFlat(array);
        assertArrayEquals(expected, actual);
    }

    @Test
    void getElementsFlat_short1d() {
        short[] array = {1, 2, 3, 4};
        short[] expected = {1, 2, 3, 4};
        short[] actual = (short[]) ArrayUtils.getElementsFlat(array);
        assertArrayEquals(expected, actual);
    }

    @Test
    void getElementsFlat_short2d() {
        short[][] array = {{1, 2, 3, 4}, {5, 6, 7, 8}};
        short[] expected = {1, 2, 3, 4, 5, 6, 7, 8};
        short[] actual = (short[]) ArrayUtils.getElementsFlat(array);
        assertArrayEquals(expected, actual);
    }

    @Test
    void getElementsFlat_short3d() {
        short[][][] array = {{{1, 2, 3, 4}, {5, 6, 7, 8}}, {{9, 10, 11, 12}, {13, 14, 15, 16}}};
        short[] expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        short[] actual = (short[]) ArrayUtils.getElementsFlat(array);
        assertArrayEquals(expected, actual);
    }

    @Test
    void getElementsFlat_int1d() {
        int[] array = {1, 2, 3, 4};
        int[] expected = {1, 2, 3, 4};
        int[] actual = (int[]) ArrayUtils.getElementsFlat(array);
        assertArrayEquals(expected, actual);
    }

    @Test
    void getElementsFlat_int2d() {
        int[][] array = {{1, 2, 3, 4}, {5, 6, 7, 8}};
        int[] expected = {1, 2, 3, 4, 5, 6, 7, 8};
        int[] actual = (int[]) ArrayUtils.getElementsFlat(array);
        assertArrayEquals(expected, actual);
    }

    @Test
    void getElementsFlat_int3d() {
        int[][][] array = {{{1, 2, 3, 4}, {5, 6, 7, 8}}, {{9, 10, 11, 12}, {13, 14, 15, 16}}};
        int[] expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        int[] actual = (int[]) ArrayUtils.getElementsFlat(array);
        assertArrayEquals(expected, actual);
    }

    @Test
    void getElementsFlat_long1d() {
        long[] array = {1, 2, 3, 4};
        long[] expected = {1, 2, 3, 4};
        long[] actual = (long[]) ArrayUtils.getElementsFlat(array);
        assertArrayEquals(expected, actual);
    }

    @Test
    void getElementsFlat_long2d() {
        long[][] array = {{1, 2, 3, 4}, {5, 6, 7, 8}};
        long[] expected = {1, 2, 3, 4, 5, 6, 7, 8};
        long[] actual = (long[]) ArrayUtils.getElementsFlat(array);
        assertArrayEquals(expected, actual);
    }

    @Test
    void getElementsFlat_long3d() {
        long[][][] array = {{{1, 2, 3, 4}, {5, 6, 7, 8}}, {{9, 10, 11, 12}, {13, 14, 15, 16}}};
        long[] expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        long[] actual = (long[]) ArrayUtils.getElementsFlat(array);
        assertArrayEquals(expected, actual);
    }

    @Test
    void getElementsFlat_float1d() {
        float[] array = {1, 2, 3, 4};
        float[] expected = {1, 2, 3, 4};
        float[] actual = (float[]) ArrayUtils.getElementsFlat(array);
        assertArrayEquals(expected, actual);
    }

    @Test
    void getElementsFlat_float2d() {
        float[][] array = {{1, 2, 3, 4}, {5, 6, 7, 8}};
        float[] expected = {1, 2, 3, 4, 5, 6, 7, 8};
        float[] actual = (float[]) ArrayUtils.getElementsFlat(array);
        assertArrayEquals(expected, actual);
    }

    @Test
    void getElementsFlat_float3d() {
        float[][][] array = {{{1, 2, 3, 4}, {5, 6, 7, 8}}, {{9, 10, 11, 12}, {13, 14, 15, 16}}};
        float[] expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        float[] actual = (float[]) ArrayUtils.getElementsFlat(array);
        assertArrayEquals(expected, actual);
    }

    @Test
    void getElementsFlat_double1d() {
        double[] array = {1, 2, 3, 4};
        double[] expected = {1, 2, 3, 4};
        double[] actual = (double[]) ArrayUtils.getElementsFlat(array);
        assertArrayEquals(expected, actual);
    }


    @Test
    void getElementsFlat_double2d() {
        double[][] array = {{1, 2, 3, 4}, {5, 6, 7, 8}};
        double[] expected = {1, 2, 3, 4, 5, 6, 7, 8};
        double[] actual = (double[]) ArrayUtils.getElementsFlat(array);
        assertArrayEquals(expected, actual);
    }

    @Test
    void getElementsFlat_double3d() {
        double[][][] array = {{{1, 2, 3, 4}, {5, 6, 7, 8}}, {{9, 10, 11, 12}, {13, 14, 15, 16}}};
        double[] expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        double[] actual = (double[]) ArrayUtils.getElementsFlat(array);
        assertArrayEquals(expected, actual);
    }

    @Test
    void getComponentType() {
    }

    @Test
    void contains() {
    }

    @Test
    void endsWidth() {
    }
}