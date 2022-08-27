package me.mikex86.scicore.utils;

import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Arrays;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class ShapeUtilsTest {

    Stream<Arguments> getNumElementsData() {
        return Stream.of(
                // shape, expected num elements
                Arguments.of(new long[]{1}, 1L),
                Arguments.of(new long[]{4}, 4L),
                Arguments.of(new long[]{8}, 8L),
                Arguments.of(new long[]{16}, 16L),
                Arguments.of(new long[]{3, 4}, 3L * 4L),
                Arguments.of(new long[]{4, 5}, 4L * 5L),
                Arguments.of(new long[]{5, 6}, 5L * 6L),
                Arguments.of(new long[]{6, 7}, 6L * 7L),
                Arguments.of(new long[]{3, 4, 5}, 3L * 4L * 5L),
                Arguments.of(new long[]{4, 5, 6}, 4L * 5L * 6L),
                Arguments.of(new long[]{5, 6, 7}, 5L * 6L * 7L),
                Arguments.of(new long[]{6, 7, 8}, 6L * 7L * 8L),
                Arguments.of(new long[]{3, 4, 5, 6}, 3L * 4L * 5L * 6L),
                Arguments.of(new long[]{4, 5, 6, 7}, 4L * 5L * 6L * 7L),
                Arguments.of(new long[]{5, 6, 7, 8}, 5L * 6L * 7L * 8L),
                Arguments.of(new long[]{3, 4, 5, 6, 7}, 3L * 4L * 5L * 6L * 7L),
                Arguments.of(new long[]{4, 5, 6, 7, 8}, 4L * 5L * 6L * 7L * 8L),
                Arguments.of(new long[]{5, 6, 7, 8, 9}, 5L * 6L * 7L * 8L * 9L),
                Arguments.of(new long[]{3, 4, 5, 6, 7, 8}, 3L * 4L * 5L * 6L * 7L * 8L),
                Arguments.of(new long[]{4, 5, 6, 7, 8, 9}, 4L * 5L * 6L * 7L * 8L * 9L),
                Arguments.of(new long[]{5, 6, 7, 8, 9, 10}, 5L * 6L * 7L * 8L * 9L * 10L)
        );
    }

    @ParameterizedTest
    @MethodSource("getNumElementsData")
    void getNumElements(long[] shape, long expectedNumElements) {
        long actual = ShapeUtils.getNumElements(shape);
        assertEquals(expectedNumElements, actual);
    }

    Stream<Arguments> getMakeStridesData() {
        return Stream.of(
                // shape, expected strides
                Arguments.of(new long[]{10}, new long[]{1}),
                Arguments.of(new long[]{3}, new long[]{1}),
                Arguments.of(new long[]{27}, new long[]{1}),
                Arguments.of(new long[]{3, 3}, new long[]{3, 1}),
                Arguments.of(new long[]{10, 10, 10}, new long[]{100, 10, 1}),
                Arguments.of(new long[]{16, 32, 8}, new long[]{32 * 8, 8, 1})
        );
    }

    @ParameterizedTest
    @MethodSource("getMakeStridesData")
    void makeStrides(long[] shape, long[] expectedStrides) {
        long[] strides = ShapeUtils.makeStrides(shape);
        assertArrayEquals(expectedStrides, strides);
    }

    Stream<Arguments> getFlatIndexData() {
        return Stream.of(
                // shape, nd-index, expected flat index
                Arguments.of(new long[]{10}, new long[]{5}, 5),
                Arguments.of(new long[]{4}, new long[]{1}, 1),
                Arguments.of(new long[]{3, 4}, new long[]{1, 1}, 4L + 1L),
                Arguments.of(new long[]{3, 4}, new long[]{2, 3}, 2 * 4L + 3L),
                Arguments.of(new long[]{3, 3}, new long[]{2, 2}, 2 * 3L + 2L),
                Arguments.of(new long[]{4, 2, 4}, new long[]{1, 2, 3}, 2 * 4 + 2 * 4L + 3L),
                Arguments.of(new long[]{4, 3, 4, 2}, new long[]{3, 2, 3, 1}, 3 * 4 * 2 * 3L + 4 * 2 * 2L + 2 * 3L + 1L)
        );
    }

    @ParameterizedTest
    @MethodSource("getFlatIndexData")
    void getFlatIndex(long[] shape, long[] index, long expectedFlatIndex) {
        long[] strides = ShapeUtils.makeStrides(shape);
        long flatIndex = ShapeUtils.getFlatIndex(index, strides);
        assertEquals(expectedFlatIndex, flatIndex);
    }

    Stream<Arguments> getTestToStringData() {
        return Stream.of(
                // shape, expected to-string
                Arguments.of(new long[]{10}, "(10)"),
                Arguments.of(new long[]{10, 20}, "(10, 20)"),
                Arguments.of(new long[]{10, 20, 30}, "(10, 20, 30)"),
                Arguments.of(new long[]{10, 20, 30, 40}, "(10, 20, 30, 40)")
        );
    }

    @ParameterizedTest
    @MethodSource("getTestToStringData")
    void testToString(long[] shape, String expectedToString) {
        String toString = ShapeUtils.toString(shape);
        assertEquals(expectedToString, toString);
    }

    Stream<Arguments> testEqualsData() {
        return Stream.of(
                // shapeA, shapeB, expected equal state (true/false)
                Arguments.of(new long[]{1}, new long[]{1}, true),
                Arguments.of(new long[]{2}, new long[]{1}, false),
                Arguments.of(new long[]{3, 4}, new long[]{5, 6}, false),
                Arguments.of(new long[]{7, 7}, new long[]{7, 7}, true),
                Arguments.of(new long[]{1, 1, 1}, new long[]{1, 1, 1}, true),
                Arguments.of(new long[]{1, 2, 3}, new long[]{1, 2, 3}, true),
                Arguments.of(new long[]{1, 1, 1}, new long[]{1, 1, 2}, false)
        );
    }

    @ParameterizedTest
    @MethodSource("testEqualsData")
    void testEquals(long[] shapeA, long[] shapeB, boolean expectedEqual) {
        assertEquals(expectedEqual, ShapeUtils.equals(shapeA, shapeB));
    }

    Stream<Arguments> incrementIndexData() {
        return Stream.of(
                // shape, prevIndex, nextIndex
                Arguments.of(new long[]{10}, new long[]{1}, new long[]{2}),
                Arguments.of(new long[]{10}, new long[]{9}, new long[]{0}),
                Arguments.of(new long[]{3, 4}, new long[]{1, 1}, new long[]{1, 2}),
                Arguments.of(new long[]{3, 4}, new long[]{1, 3}, new long[]{2, 0}),
                Arguments.of(new long[]{2, 2}, new long[]{1, 1}, new long[]{0, 0})
        );
    }

    @ParameterizedTest
    @MethodSource("incrementIndexData")
    void incrementIndex(long[] shape, long[] prevIndex, long[] expectedNextIndex) {
        long[] index = Arrays.copyOf(prevIndex, prevIndex.length);
        ShapeUtils.incrementIndex(shape, index);
        assertArrayEquals(expectedNextIndex, index);
    }

    Stream<Arguments> broadcastShapes_successData() {
        return Stream.of(
                // shapeA, shapeB, expected broadcast-ed shape
                Arguments.of(new long[]{256, 256, 3}, new long[]{3}, new long[]{256, 256, 3}),
                Arguments.of(new long[]{8, 1, 6, 1}, new long[]{7, 1, 5}, new long[]{8, 7, 6, 5}),
                Arguments.of(new long[]{5, 4}, new long[]{1}, new long[]{5, 4}),
                Arguments.of(new long[]{5, 4}, new long[]{4}, new long[]{5, 4}),
                Arguments.of(new long[]{15, 3, 5}, new long[]{15, 1, 5}, new long[]{15, 3, 5}),
                Arguments.of(new long[]{15, 1, 5}, new long[]{15, 3, 5}, new long[]{15, 3, 5}),
                Arguments.of(new long[]{15, 3, 5}, new long[]{3, 5}, new long[]{15, 3, 5}),
                Arguments.of(new long[]{15, 3, 5}, new long[]{3, 1}, new long[]{15, 3, 5}),
                Arguments.of(new long[]{2, 3, 4}, new long[]{1, 2, 3, 4}, new long[]{1, 2, 3, 4}),
                Arguments.of(new long[]{1, 2, 3}, new long[]{3, 4, 1, 2, 3}, new long[]{3, 4, 1, 2, 3})
        );
    }

    @ParameterizedTest
    @MethodSource("broadcastShapes_successData")
    void broadcastShapes_success(long[] shapeA, long[] shapeB, long[] expectedBroadcast) {
        long[] broadcastedShape = ShapeUtils.broadcastShapes(shapeA, shapeB);
        assertArrayEquals(expectedBroadcast, broadcastedShape);
    }

    Stream<Arguments> broadcastShapes_failureData() {
        return Stream.of(
                // shapeA, shapeB
                Arguments.of(new long[]{3}, new long[]{4}),
                Arguments.of(new long[]{2, 3, 4}, new long[]{2, 3, 3}),
                Arguments.of(new long[]{2, 3, 4}, new long[]{3, 3, 4}),
                Arguments.of(new long[]{2, 1}, new long[]{8, 4, 3})
        );
    }

    @ParameterizedTest
    @MethodSource("broadcastShapes_failureData")
    void broadcastShapes_failure(long[] shapeA, long[] shapeB) {
        assertThrows(IllegalArgumentException.class, () -> ShapeUtils.broadcastShapes(shapeA, shapeB));
    }

    @ParameterizedTest
    @MethodSource("testIsScalarData")
    public void testIsScalar(long @NotNull [] shape, boolean expectedState) {
        assertEquals(expectedState, ShapeUtils.isScalar(shape));
    }

    @NotNull
    public Stream<Arguments> testIsScalarData() {
        return Stream.of(
                Arguments.of(new long[0], true),
                Arguments.of(new long[]{1}, true),
                Arguments.of(new long[]{1, 1}, true),
                Arguments.of(new long[]{1, 1, 1}, true),
                Arguments.of(new long[]{1, 1, 1, 1}, true),
                Arguments.of(new long[]{1, 1, 1, 1, 1}, true),
                Arguments.of(new long[]{2}, false),
                Arguments.of(new long[]{2, 2}, false),
                Arguments.of(new long[]{2, 3, 1}, false),
                Arguments.of(new long[]{1, 3, 1, 1}, false)
        );
    }

    Stream<Arguments> constrainIndexData() {
        return Stream.of(
                // index, shape, expected constrained index
                Arguments.of(new long[]{15, 32, 4}, new long[]{3, 4, 5}, new long[]{15 % 3, 32 % 4, 4 % 5}),
                Arguments.of(new long[]{4, 4, 4}, new long[]{10, 20, 30}, new long[]{4, 4, 4})
        );
    }

    @ParameterizedTest
    @MethodSource("constrainIndexData")
    void constrainIndex(long[] index, long[] shape, long[] expectedConstrainedIndex) {
        ShapeUtils.constrainIndex(index, shape);
        assertArrayEquals(expectedConstrainedIndex, index);
    }

    Stream<Arguments> getArrayShapeData() {
        return Stream.of(
                Arguments.of(new byte[4][8], new long[]{4, 8}),
                Arguments.of(new short[8][9], new long[]{8, 9}),
                Arguments.of(new int[10][20][30], new long[]{10, 20, 30}),
                Arguments.of(new long[9][7][2][1], new long[]{9, 7, 2, 1}),
                Arguments.of(new float[7][3][9], new long[]{7, 3, 9}),
                Arguments.of(new double[4][3][1][67], new long[]{4, 3, 1, 67})
        );
    }

    @ParameterizedTest
    @MethodSource("getArrayShapeData")
    void getArrayShape(Object array, long[] expectedShape) {
        long[] shape = ShapeUtils.getArrayShape(array);
        assertArrayEquals(expectedShape, shape);
    }

    @ParameterizedTest
    @MethodSource("getCommonShapeData")
    void getCommonShape(long[] shapeA, long[] shapeB, long[] expectedCommonShape) {
        long[] commonShape = ShapeUtils.getCommonShape(shapeA, shapeB);
        assertArrayEquals(expectedCommonShape, commonShape);
    }

    Stream<Arguments> getCommonShapeData() {
        return Stream.of(
                Arguments.of(new long[]{1, 2, 3}, new long[]{4, 5, 6}, new long[]{}),
                Arguments.of(new long[]{1, 1}, new long[]{1}, new long[]{1}),
                Arguments.of(new long[]{1, 1}, new long[]{1, 1, 1}, new long[]{1, 1}),
                Arguments.of(new long[]{1, 1, 1}, new long[]{1, 1, 1}, new long[]{1, 1, 1}),
                Arguments.of(new long[]{2, 1, 1}, new long[]{1, 1, 1}, new long[]{1, 1}),
                Arguments.of(new long[]{1, 3, 2}, new long[]{1, 3, 2}, new long[]{1, 3, 2}),
                Arguments.of(new long[]{1, 1, 1}, new long[]{1, 1, 1, 1}, new long[]{1, 1, 1})
        );
    }

    @ParameterizedTest
    @MethodSource("compareBroadcastRank_successData")
    void compareBroadcastRank_success(long @NotNull [] shapeA, long @NotNull [] shapeB, int expectedResult) {
        assertEquals(expectedResult, ShapeUtils.compareBroadcastRank(shapeA, shapeB));
    }

    Stream<Arguments> compareBroadcastRank_successData() {
        return Stream.of(
                Arguments.of(new long[]{1}, new long[]{1, 1}, -1),
                Arguments.of(new long[]{1, 1}, new long[]{1, 1}, 0),
                Arguments.of(new long[]{1, 1}, new long[]{1}, 1),

                Arguments.of(new long[]{2, 1}, new long[]{1, 1}, 1),
                Arguments.of(new long[]{1, 1}, new long[]{2, 1}, -1),
                Arguments.of(new long[]{2, 1}, new long[]{2, 1}, 0),

                Arguments.of(new long[]{1, 1, 1}, new long[]{1, 1, 1}, 0),
                Arguments.of(new long[]{1, 1, 1}, new long[]{1, 1, 1, 1}, -1),
                Arguments.of(new long[]{1, 1, 1}, new long[]{1, 1, 1, 1, 1}, -1),

                Arguments.of(new long[]{1, 2, 3}, new long[]{2, 2, 3}, -1),
                Arguments.of(new long[]{2, 2, 3}, new long[]{1, 2, 3}, 1),
                Arguments.of(new long[]{2, 2, 3}, new long[]{2, 2, 3}, 0),
                Arguments.of(new long[]{2, 2, 3}, new long[]{3, 2, 2, 3}, -1),
                Arguments.of(new long[]{3, 2, 2, 3}, new long[]{2, 2, 3}, 1),

                Arguments.of(new long[]{1, 1, 1, 1}, new long[]{1, 1, 1}, 1),
                Arguments.of(new long[]{1, 1, 1, 1}, new long[]{1, 1, 1, 1}, 0),
                Arguments.of(new long[]{1, 1, 1, 1}, new long[]{1, 1, 1, 1, 1}, -1),
                Arguments.of(new long[]{2}, new long[]{1, 1, 1, 2}, -1),

                Arguments.of(new long[]{1, 2, 2, 2}, new long[]{2, 2, 2, 2}, -1),
                Arguments.of(new long[]{2, 2, 2, 2}, new long[]{1, 2, 2, 2}, 1),
                Arguments.of(new long[]{2, 2, 2, 2}, new long[]{2, 2, 2, 2}, 0),
                Arguments.of(new long[]{2, 2, 3, 4}, new long[]{1, 2, 3, 4}, 1)
        );
    }


    @ParameterizedTest
    @MethodSource("compareBroadcastRank_failureData")
    void compareBroadcastRank_failure(long @NotNull [] shapeA, long @NotNull [] shapeB) {
        assertThrows(IllegalArgumentException.class, () -> ShapeUtils.compareBroadcastRank(shapeA, shapeB));
    }

    Stream<Arguments> compareBroadcastRank_failureData() {
        return Stream.of(
                Arguments.of(new long[]{2, 3, 4}, new long[]{1, 2, 3}),
                Arguments.of(new long[]{1, 2, 3}, new long[]{2, 3, 4}),
                Arguments.of(new long[]{3, 3, 4}, new long[]{1, 2, 3, 4})
        );
    }
}