package me.mikex86.scicore;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ITensorTest {

    private static final float EPSILON = 1E-6f;

    SciCore sciCore;

    @BeforeEach
    void setUp() {
        sciCore = new SciCore();
        sciCore.setBackend(SciCore.BackendType.JVM);
    }

    @Test
    void getView() {
        ITensor matrix = sciCore.matrix(new float[][]{{12.4f, 16.3f}, {1.2f, 9.1f}, {7.3f, 3.4f}});

        assertDoesNotThrow(() -> matrix.getView(0));
        assertDoesNotThrow(() -> matrix.getView(1));
        assertDoesNotThrow(() -> matrix.getView(2));
        assertThrows(IndexOutOfBoundsException.class, () -> matrix.getView(3));

        ITensor view1 = matrix.getView(0);
        assertEquals(12.4f, view1.getFloat(0));
        assertEquals(16.3f, view1.getFloat(1));
        assertThrows(IndexOutOfBoundsException.class, () -> view1.getFloat(2));

        ITensor view2 = matrix.getView(1);
        assertEquals(1.2f, view2.getFloat(0));
        assertEquals(9.1f, view2.getFloat(1));
        assertThrows(IndexOutOfBoundsException.class, () -> view2.getFloat(2));

        matrix.setFloat(42.0f, 1, 1);
        assertEquals(42.0f, view2.getFloat(1));
    }

    @Test
    void copy() {
        ITensor array = sciCore.array(new float[]{1.0f, 2.0f, 3.0f, 4.0f});
        ITensor copy = array.copy();
        assertEquals(array.getFloat(0), copy.getFloat(0));
        assertEquals(array.getFloat(1), copy.getFloat(1));
        assertEquals(array.getFloat(2), copy.getFloat(2));
        assertEquals(array.getFloat(3), copy.getFloat(3));
    }

    @Test
    void exp() {
        ITensor matrix = sciCore.matrix(new float[][]{{3.8f, 46.3f}, {2.7f, 1.9f}, {3.7f, 1.7f}});
        ITensor exp = matrix.exp();
        assertEquals((float) Math.exp(3.8f), exp.getFloat(0, 0), EPSILON);
        assertEquals((float) Math.exp(46.3f), exp.getFloat(0, 1), EPSILON);
    }

    @Test
    void divided_test_2x2x2by2x2() {
        ITensor a = sciCore.ndarray(new float[][][]{{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}});
        ITensor b = sciCore.matrix(new float[][]{{5.0f, 6.0f}, {7.0f, 8.0f}});
        ITensor result = a.divided(b);
        assertEquals(1.0f / 5.0f, result.getFloat(0, 0, 0), EPSILON);
        assertEquals(2.0f / 6.0f, result.getFloat(0, 0, 1), EPSILON);
        assertEquals(3.0f / 7.0f, result.getFloat(0, 1, 0), EPSILON);
        assertEquals(4.0f / 8.0f, result.getFloat(0, 1, 1), EPSILON);
        assertEquals(5.0f / 5.0f, result.getFloat(1, 0, 0), EPSILON);
        assertEquals(6.0f / 6.0f, result.getFloat(1, 0, 1), EPSILON);
        assertEquals(7.0f / 7.0f, result.getFloat(1, 1, 0), EPSILON);
        assertEquals(8.0f / 8.0f, result.getFloat(1, 1, 1), EPSILON);
    }

    @Test
    void divided_test_2x2by2x1() {
        ITensor a = sciCore.matrix(new double[][]{{4.4701180e+01, 1.2818411e+20}, {1.4879732e+01, 6.6858945e+00}});
        ITensor b = sciCore.matrix(new double[][]{{1.2818411e+20}, {2.1565626e+01}});
        ITensor result = a.divided(b);
        assertEquals(3.4872637e-19, result.getDouble(0, 0), EPSILON);
        assertEquals(1.0000000e+00, result.getDouble(0, 1), EPSILON);
        assertEquals(6.8997449e-01, result.getDouble(1, 0), EPSILON);
        assertEquals(3.1002551e-01, result.getDouble(1, 1), EPSILON);
    }

    @Test
    void softmax_dim1() {
        ITensor matrix = sciCore.matrix(new float[][]{{3.8f, 46.3f}, {2.7f, 1.9f}});
        ITensor softmax = matrix.softmax(1);
        assertEquals(3.4872616e-19, softmax.getFloat(0, 0), EPSILON);
        assertEquals(1.0000000e+00, softmax.getFloat(0, 1), EPSILON);
        assertEquals(6.8997449e-01, softmax.getFloat(1, 0), EPSILON);
        assertEquals(3.1002548e-01, softmax.getFloat(1, 1), EPSILON);
    }

    @Test
    void reduceSum_1x10_dim_minusOne_noKeepDims() {
        ITensor matrix = sciCore.matrix(new float[][]{{3.8f, 3.35f, 81.3f, 39.1f, 9.3f, 1.9f}});
        ITensor reduced = matrix.reduceSum(-1);
        assertEquals(138.75f, reduced.getFloat(0), EPSILON);
    }

    @Test
    void reduceSum_3x3_dim0_noKeepDims() {
        ITensor matrix = sciCore.matrix(new float[][]{{3.0f, 1.0f, 4.0f}, {7.0f, 8.0f, 2.0f}, {11.0f, 2.0f, 1.0f}});
        ITensor sum = matrix.reduceSum(0);
        assertArrayEquals(new long[]{3}, sum.getShape());
        assertEquals(21.0f, sum.getFloat(0), EPSILON);
        assertEquals(11.0f, sum.getFloat(1), EPSILON);
        assertEquals(7.0f, sum.getFloat(2), EPSILON);
    }

    @Test
    void reduceSum_4x3_dim0_noKeepDims() {
        ITensor matrix = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}, {10.0f, 11.0f, 12.0f}});
        ITensor sum = matrix.reduceSum(0);
        assertArrayEquals(new long[]{3}, sum.getShape());
        assertEquals(22.0f, sum.getFloat(0), EPSILON);
        assertEquals(26.0f, sum.getFloat(1), EPSILON);
        assertEquals(30.0f, sum.getFloat(2), EPSILON);
    }

    @Test
    void reduceSum4x3_dim1_noKeepDims() {
        ITensor matrix = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}, {10.0f, 11.0f, 12.0f}});
        ITensor sum = matrix.reduceSum(1);
        assertArrayEquals(new long[]{4}, sum.getShape());
        assertEquals(6.0f, sum.getFloat(0), EPSILON);
        assertEquals(15.0f, sum.getFloat(1), EPSILON);
        assertEquals(24.0f, sum.getFloat(2), EPSILON);
        assertEquals(33.0f, sum.getFloat(3), EPSILON);
    }

    @Test
    void reduceSum_5x3x4_noKeepDims() {
        ITensor tensor = sciCore.random(DataType.FLOAT32, 5, 3, 4);
        ITensor sum = tensor.reduceSum(1);
        assertArrayEquals(new long[]{5, 4}, sum.getShape());
    }
}