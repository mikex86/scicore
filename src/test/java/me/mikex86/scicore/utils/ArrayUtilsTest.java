package me.mikex86.scicore.utils;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
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

    Stream<Arguments> getComponentTypeData() {
        return Stream.of(
                // array, expected component type
                Arguments.of(new byte[1238], byte.class),
                Arguments.of(new short[38][2][3][1][2], short.class),
                Arguments.of(new int[1], int.class),
                Arguments.of(new int[4][3][7][8], int.class),
                Arguments.of(new long[3][2][1], long.class),
                Arguments.of(new float[2][2], float.class),
                Arguments.of(new float[1], float.class),
                Arguments.of(new double[1], double.class),
                Arguments.of(new double[2][2], double.class)
        );
    }

    @ParameterizedTest
    @MethodSource("getComponentTypeData")
    void getComponentType(Object array, Class<?> expectedComponentType) {
        Class<?> componentType = ArrayUtils.getComponentType(array);
        assertEquals(expectedComponentType, componentType);
    }

    Stream<Arguments> containsData() {
        return Stream.of(
                // arrayA, arrayB, expected contains
                Arguments.of(new long[]{1, 2, 3, 4, 5, 6, 7}, new long[]{1}, true),
                Arguments.of(new long[]{1, 2, 3, 4, 5, 6, 7}, new long[]{1, 2, 3}, true),
                Arguments.of(new long[]{1, 2, 3, 4, 5, 6, 7}, new long[]{1}, true),
                Arguments.of(new long[]{1, 2, 3, 4, 5, 6, 7}, new long[]{0}, false),
                Arguments.of(new long[]{1, 2, 3, 4, 5, 6, 7}, new long[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, false),
                Arguments.of(new long[]{1, 2, 3, 4, 5, 6, 7}, new long[]{4, 5, 6}, true),
                Arguments.of(new long[]{1, 2, 3, 4, 5, 6, 7}, new long[]{5, 6, 7}, true),
                Arguments.of(new long[]{1, 2, 3, 4, 5, 6, 7}, new long[]{5, 6, 6}, false),
                Arguments.of(new long[]{7, 3, 1}, new long[]{1, 3, 3, 7}, false)
        );
    }

    @ParameterizedTest
    @MethodSource("containsData")
    void contains(long[] arrayA, long[] arrayB, boolean expectedContains) {
        boolean contains = ArrayUtils.contains(arrayA, arrayB);
        assertEquals(expectedContains, contains);
    }

    Stream<Arguments> endsWithData() {
        return Stream.of(
                // arrayA, arrayB, expected ends with
                Arguments.of(new long[]{1, 2, 3, 4, 5, 6, 7}, new long[]{5, 6, 7}, true),
                Arguments.of(new long[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, new long[]{1, 2, 3, 4}, false),
                Arguments.of(new long[]{1, 2, 3, 4, 5, 6}, new long[]{6}, true),
                Arguments.of(new long[]{1, 2, 3, 4, 5, 6, 7}, new long[]{4, 5, 6}, false),
                Arguments.of(new long[]{1, 2, 3, 4, 5, 6}, new long[]{2, 4, 5, 6}, false)
        );
    }

    @ParameterizedTest
    @MethodSource("endsWithData")
    void endsWidth(long[] arrayA, long[] arrayB, boolean expectedEndsWith) {
        boolean endsWidth = ArrayUtils.endsWidth(arrayA, arrayB);
        assertEquals(expectedEndsWith, endsWidth);
    }
}