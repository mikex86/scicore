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
    void softmax() {
        ITensor matrix = sciCore.matrix(new float[][]{{3.8f, 46.3f}, {2.7f, 1.9f}});
        //ITensor softmax = matrix.softmax(1);
        //assertEquals(3.4872616e-19, softmax.getFloat(0, 0), EPSILON);
        //assertEquals(1.0000000e+00, softmax.getFloat(0, 1), EPSILON);
        //assertEquals(6.8997449e-01, softmax.getFloat(1, 0), EPSILON);
        //assertEquals(3.1002548e-01, softmax.getFloat(1, 1), EPSILON);
    }

    @Test
    void reduceSum_3x3_dim0() {
        ITensor matrix = sciCore.matrix(new float[][] {{3.0f, 1.0f, 4.0f}, {7.0f, 8.0f, 2.0f}, {11.0f, 2.0f, 1.0f}});
        ITensor sum = matrix.reduceSum(0);
        assertArrayEquals(new long[]{3}, sum.getShape());
//        assertEquals(21.0f, sum.getFloat(0), EPSILON);
//        assertEquals(27.0f, sum.getFloat(1), EPSILON);
//        assertEquals(7.0f, sum.getFloat(2), EPSILON);
    }

    @Test
    void reduceSum_4x3_dim0() {
        ITensor matrix = sciCore.matrix(new float[][] {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}, {10.0f, 11.0f, 12.0f}});
        ITensor sum = matrix.reduceSum(0);
        assertArrayEquals(new long[]{3}, sum.getShape());
        assertEquals(22.0f, sum.getFloat(0), EPSILON);
        assertEquals(26.0f, sum.getFloat(1), EPSILON);
        assertEquals(30.0f, sum.getFloat(2), EPSILON);
    }

//    @Test
//    void reduceSum_5x3x4() {
//        ITensor tensor = sciCore.random(DataType.FLOAT32, 5, 3, 4);
//        ITensor sum = tensor.reduceSum(1);
//        assertArrayEquals(new long[]{5, 4}, sum.getShape());
//    }
}