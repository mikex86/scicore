package me.mikex86.scicore.utils;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class ShapeUtilsTest {

    Stream<Arguments> getNumElementsData() {
        return Stream.of(
                Arguments.of(1L, new long[]{1}),
                Arguments.of(4L, new long[]{4}),
                Arguments.of(8L, new long[]{8}),
                Arguments.of(16L, new long[]{16}),
                Arguments.of(3L * 4L, new long[]{3, 4}),
                Arguments.of(4L * 5L, new long[]{4, 5}),
                Arguments.of(5L * 6L, new long[]{5, 6}),
                Arguments.of(6L * 7L, new long[]{6, 7}),
                Arguments.of(3L * 4L * 5L, new long[]{3, 4, 5}),
                Arguments.of(4L * 5L * 6L, new long[]{4, 5, 6}),
                Arguments.of(5L * 6L * 7L, new long[]{5, 6, 7}),
                Arguments.of(6L * 7L * 8L, new long[]{6, 7, 8}),
                Arguments.of(3L * 4L * 5L * 6L, new long[]{3, 4, 5, 6}),
                Arguments.of(4L * 5L * 6L * 7L, new long[]{4, 5, 6, 7}),
                Arguments.of(5L * 6L * 7L * 8L, new long[]{5, 6, 7, 8}),
                Arguments.of(3L * 4L * 5L * 6L * 7L, new long[]{3, 4, 5, 6, 7}),
                Arguments.of(4L * 5L * 6L * 7L * 8L, new long[]{4, 5, 6, 7, 8}),
                Arguments.of(5L * 6L * 7L * 8L * 9L, new long[]{5, 6, 7, 8, 9}),
                Arguments.of(3L * 4L * 5L * 6L * 7L * 8L, new long[]{3, 4, 5, 6, 7, 8}),
                Arguments.of(4L * 5L * 6L * 7L * 8L * 9L, new long[]{4, 5, 6, 7, 8, 9}),
                Arguments.of(5L * 6L * 7L * 8L * 9L * 10L, new long[]{5, 6, 7, 8, 9, 10})
        );
    }

    @ParameterizedTest
    @MethodSource("getNumElementsData")
    void getNumElements(long expectedNumElements, long[] shape) {
        long actual = ShapeUtils.getNumElements(shape);
        assertEquals(expectedNumElements, actual);
    }

    @Test
    void makeStrides() {
    }

    @Test
    void getFlatIndex() {
    }

    @Test
    void testToString() {
    }

    @Test
    void testEquals() {
    }

    @Test
    void incrementIndex() {
    }

    @Test
    void broadcastShapes() {
    }

    @Test
    void getArrayShape() {
    }

}