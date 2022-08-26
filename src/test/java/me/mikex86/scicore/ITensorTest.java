package me.mikex86.scicore;

import me.mikex86.scicore.utils.ShapeUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

@TestInstance(TestInstance.Lifecycle.PER_METHOD)
class ITensorTest {

    private static final float EPSILON = 1E-6f;

    ISciCore sciCore;

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

    static Stream<Object> testNdArrayShapeData() {
        return Stream.of(
                // nd-shaped java array
                new byte[1],
                new byte[1][1][1],
                new byte[1][1][1][1],
                new short[1],
                new short[1][1][1][1][1][1],
                new short[2][3][4][5],
                new float[12][12][12],
                new float[1][1][1],
                new float[4][4][4],
                new double[1][4][1]
        );
    }

    @ParameterizedTest
    @MethodSource("testNdArrayShapeData")
    void testNdArrayShape(Object javaArray) {
        long[] shape = ShapeUtils.getArrayShape(javaArray);
        ITensor ndArray = sciCore.ndarray(javaArray);
        assertArrayEquals(shape, ndArray.getShape());
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
    void toString_test_scalar() {
        ITensor scalar = sciCore.scalar(42.0f);
        assertEquals("JvmTensor(dtype=FLOAT32, shape=[], isScalar=true, data=42.0)", scalar.toString());
    }

    @Test
    void toString_test_1dArray() {
        ITensor array = sciCore.array(new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
        assertEquals("JvmTensor(dtype=FLOAT32, shape=[5], data=[1.0, 2.0, 3.0, 4.0, 5.0])", array.toString());
    }

    @Test
    void toString_test_2dMatrix() {
        ITensor matrix = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
        assertEquals("JvmTensor(dtype=FLOAT32, shape=[2, 3], data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])", matrix.toString());
    }

    @Test
    void toString_test_2dMatrix_large() {
        ITensor matrix = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f},
                {16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f}});
        assertEquals("""
                JvmTensor(dtype=FLOAT32, shape=[2, 15], data=
                \t[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                \t [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0]])""", matrix.toString());
    }

    @Test
    void exp() {
        ITensor matrix = sciCore.matrix(new float[][]{{3.8f, 46.3f}, {2.7f, 1.9f}, {3.7f, 1.7f}});
        ITensor exp = matrix.exp();
        assertEquals((float) Math.exp(3.8f), exp.getFloat(0, 0), EPSILON);
        assertEquals((float) Math.exp(46.3f), exp.getFloat(0, 1), EPSILON);
    }

    @Test
    void matmul_test_2x2by2x2() {
        ITensor matrixA = sciCore.matrix(new float[][]{{1, 2}, {3, 4}});
        ITensor matrixB = sciCore.matrix(new float[][]{{5, 6}, {7, 8}});
        ITensor result = matrixA.matmul(matrixB);
        assertEquals(sciCore.matrix(new float[][]{{19, 22}, {43, 50}}), result);
    }

    @Test
    void matmul_test_2x3by2x3_failure() {
        ITensor matrixA = sciCore.matrix(new float[][]{{1, 2, 3}, {4, 5, 6}});
        ITensor matrixB = sciCore.matrix(new float[][]{{7, 8, 9}, {10, 11, 12}});
        assertThrows(IllegalArgumentException.class, () -> matrixA.matmul(matrixB));
    }

    @Test
    void matmul_test_3d_failure() {
        ITensor matrixA = sciCore.ndarray(new float[3][4][5]);
        ITensor matrixB = sciCore.ndarray(new float[8][9][10]);
        assertThrows(IllegalArgumentException.class, () -> matrixA.matmul(matrixB));
    }

    @Test
    void matmul_test_2x3by3x2() {
        ITensor matrixA = sciCore.matrix(new float[][]{{1, 2, 3}, {4, 5, 6}});
        ITensor matrixB = sciCore.matrix(new float[][]{{7, 8}, {9, 10}, {11, 12}});
        ITensor result = matrixA.matmul(matrixB);
        assertEquals(sciCore.matrix(new float[][]{{58, 64}, {139, 154}}), result);
    }

    @Test
    void matmul_test_withDimView() {
        ITensor bigNdArrayA = sciCore.ndarray(new float[][][]{
                {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                {{10, 11, 12}, {13, 14, 15}, {16, 17, 18}}
        });
        ITensor matrixA = bigNdArrayA.getView(0);
        ITensor matrixB = bigNdArrayA.getView(1);

        assertEquals(sciCore.matrix(new float[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}), matrixA);
        assertEquals(sciCore.matrix(new float[][]{{10, 11, 12}, {13, 14, 15}, {16, 17, 18}}), matrixB);

        ITensor result = matrixA.matmul(matrixB);
        assertEquals(sciCore.matrix(new float[][]{{84, 90, 96}, {201, 216, 231}, {318, 342, 366}}), result);
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
    void softmax_test_dim1() {
        ITensor matrix = sciCore.matrix(new float[][]{{3.8f, 46.3f}, {2.7f, 1.9f}});
        ITensor softmax = matrix.softmax(1);
        assertEquals(sciCore.matrix(new float[][]{{3.4872616e-19f, 1.0000000e+00f}, {6.8997449e-01f, 3.1002548e-01f}}), softmax);
    }

    @Test
    void reduceSum_test_1x10_dim_minusOne_noKeepDims() {
        ITensor matrix = sciCore.matrix(new float[][]{{3.8f, 3.35f, 81.3f, 39.1f, 9.3f, 1.9f}});
        ITensor reduced = matrix.reduceSum(-1);
        assertEquals(sciCore.scalar(3.8f + 3.35f + 81.3f + 39.1f + 9.3f + 1.9f), reduced);
    }

    @Test
    void reduceSum_test_1x10_dim_minusOne_keepDims() {
        ITensor matrix = sciCore.matrix(new float[][]{{3.8f, 3.35f, 81.3f, 39.1f, 9.3f, 1.9f}});
        ITensor reduced = matrix.reduceSum(-1, true);
        assertEquals(sciCore.matrix(new float[][]{{3.8f + 3.35f + 81.3f + 39.1f + 9.3f + 1.9f}}), reduced);
    }

    @Test
    void reduceSum_test_1x10_dim0_noKeepDims() {
        ITensor matrix = sciCore.matrix(new float[][]{{3.8f, 3.35f, 81.3f, 39.1f, 9.3f, 1.9f}});
        ITensor reduced = matrix.reduceSum(0);
        assertEquals(sciCore.array(new float[]{3.8f, 3.35f, 81.3f, 39.1f, 9.3f, 1.9f}), reduced);
    }

    @Test
    void reduceSum_test_1x10_dim0_keepDims() {
        ITensor matrix = sciCore.matrix(new float[][]{{3.8f, 3.35f, 81.3f, 39.1f, 9.3f, 1.9f}});
        ITensor reduced = matrix.reduceSum(0, true);
        assertEquals(sciCore.matrix(new float[][]{{3.8f, 3.35f, 81.3f, 39.1f, 9.3f, 1.9f}}), reduced);
    }

    @Test
    void reduceSum_test_3x3_dim0_noKeepDims() {
        ITensor matrix = sciCore.matrix(new float[][]{{3.0f, 1.0f, 4.0f}, {7.0f, 8.0f, 2.0f}, {11.0f, 2.0f, 1.0f}});
        ITensor sum = matrix.reduceSum(0);
        assertEquals(sciCore.array(new float[]{3.0f + 7.0f + 11.0f, 1.0f + 8.0f + 2.0f, 4.0f + 2.0f + 1.0f}), sum);
    }

    @Test
    void reduceSum_test_3x3_dim0_keepDims() {
        ITensor matrix = sciCore.matrix(new float[][]{{3.0f, 1.0f, 4.0f}, {7.0f, 8.0f, 2.0f}, {11.0f, 2.0f, 1.0f}});
        ITensor sum = matrix.reduceSum(0, true);
        assertEquals(sciCore.matrix(new float[][]{{3.0f + 7.0f + 11.0f, 1.0f + 8.0f + 2.0f, 4.0f + 2.0f + 1.0f}}), sum);
    }

    @Test
    void reduceSum_test_3x3_dim1_noKeepDims() {
        ITensor matrix = sciCore.matrix(new float[][]{{3.0f, 1.0f, 4.0f}, {7.0f, 8.0f, 2.0f}, {11.0f, 2.0f, 1.0f}});
        ITensor sum = matrix.reduceSum(1);
        assertEquals(sciCore.array(new float[]{3.0f + 1.0f + 4.0f, 7.0f + 8.0f + 2.0f, 11.0f + 2.0f + 1.0f}), sum);
    }

    @Test
    void reduceSum_test_3x3_dim1_keepDims() {
        ITensor matrix = sciCore.matrix(new float[][]{{3.0f, 1.0f, 4.0f}, {7.0f, 8.0f, 2.0f}, {11.0f, 2.0f, 1.0f}});
        ITensor sum = matrix.reduceSum(1, true);
        assertEquals(sciCore.matrix(new float[][]{{3.0f + 1.0f + 4.0f}, {7.0f + 8.0f + 2.0f}, {11.0f + 2.0f + 1.0f}}), sum);
    }


    @Test
    void reduceSum_test_4x3_dim_minusOne_noKeepDims() {
        ITensor matrix = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}, {10.0f, 11.0f, 12.0f}});
        ITensor sum = matrix.reduceSum(-1);
        assertEquals(sciCore.scalar(1.0f + 2.0f + 3.0f + 4.0f + 5.0f + 6.0f + 7.0f + 8.0f + 9.0f + 10.0f + 11.0f + 12.0f), sum);
    }

    @Test
    void reduceSum_test_4x3_dim_minusOne_keepDims() {
        ITensor matrix = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}, {10.0f, 11.0f, 12.0f}});
        ITensor sum = matrix.reduceSum(-1, true);
        assertEquals(sciCore.matrix(new float[][]{{1.0f + 2.0f + 3.0f + 4.0f + 5.0f + 6.0f + 7.0f + 8.0f + 9.0f + 10.0f + 11.0f + 12.0f}}), sum);
    }

    @Test
    void reduceSum_test_4x3_dim0_noKeepDims() {
        ITensor matrix = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}, {10.0f, 11.0f, 12.0f}});
        ITensor sum = matrix.reduceSum(0);
        assertEquals(sciCore.array(new float[]{1.0f + 4.0f + 7.0f + 10.0f, 2.0f + 5.0f + 8.0f + 11.0f, 3.0f + 6.0f + 9.0f + 12.0f}), sum);
    }

    @Test
    void reduceSum_test_4x3_dim0_keepDims() {
        ITensor matrix = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}, {10.0f, 11.0f, 12.0f}});
        ITensor sum = matrix.reduceSum(0, true);
        assertEquals(sciCore.matrix(new float[][]{{1.0f + 4.0f + 7.0f + 10.0f, 2.0f + 5.0f + 8.0f + 11.0f, 3.0f + 6.0f + 9.0f + 12.0f}}), sum);
    }

    @Test
    void reduceSum_test_4x3_dim1_noKeepDims() {
        ITensor matrix = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}, {10.0f, 11.0f, 12.0f}});
        ITensor sum = matrix.reduceSum(1);
        assertEquals(sciCore.array(new float[]{1.0f + 2.0f + 3.0f, 4.0f + 5.0f + 6.0f, 7.0f + 8.0f + 9.0f, 10.0f + 11.0f + 12.0f}), sum);
    }

    @Test
    void reduceSum_test_4x3_dim1_keepDims() {
        ITensor matrix = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}, {10.0f, 11.0f, 12.0f}});
        ITensor sum = matrix.reduceSum(1, true);
        assertEquals(sciCore.matrix(new float[][]{{1.0f + 2.0f + 3.0f}, {4.0f + 5.0f + 6.0f}, {7.0f + 8.0f + 9.0f}, {10.0f + 11.0f + 12.0f}}), sum);
    }

    @Test
    void reduceSum_test_2x2x2_dim_minusOne_noKeepDims() {
        ITensor ndarray = sciCore.ndarray(new float[][][]{{{10.0f, 12.0f}, {13.0f, 14.0f}}, {{15.0f, 16.0f}, {17.0f, 18.0f}}});
        ITensor sum = ndarray.reduceSum(-1);
        assertEquals(sciCore.scalar(10.0f + 12.0f + 13.0f + 14.0f + 15.0f + 16.0f + 17.0f + 18.0f), sum);
    }

    @Test
    void reduceSum_test_2x2x2_dim_minusOne_keepDims() {
        ITensor ndarray = sciCore.ndarray(new float[][][]{{{10.0f, 12.0f}, {13.0f, 14.0f}}, {{15.0f, 16.0f}, {17.0f, 18.0f}}});
        ITensor sum = ndarray.reduceSum(-1, true);
        assertEquals(sciCore.ndarray(new float[][][]{{{10.0f + 12.0f + 13.0f + 14.0f + 15.0f + 16.0f + 17.0f + 18.0f}}}), sum);
    }

    @Test
    void reduceSum_test_2x2x2_dim0_noKeepDims() {
        ITensor ndarray = sciCore.ndarray(new float[][][]{{{10.0f, 12.0f}, {13.0f, 14.0f}}, {{15.0f, 16.0f}, {17.0f, 18.0f}}});
        ITensor sum = ndarray.reduceSum(0);
        assertEquals(sciCore.matrix(new float[][]{{10.0f + 15.0f, 12.0f + 16.0f}, {13.0f + 17.0f, 14.0f + 18.0f}}), sum);
    }

    @Test
    void reduceSum_test_2x2x2_dim0_keepDims() {
        ITensor ndarray = sciCore.ndarray(new float[][][]{{{10.0f, 12.0f}, {13.0f, 14.0f}}, {{15.0f, 16.0f}, {17.0f, 18.0f}}});
        ITensor sum = ndarray.reduceSum(0, true);
        assertEquals(sciCore.ndarray(new float[][][]{{{10.0f + 15.0f, 12.0f + 16.0f}, {13.0f + 17.0f, 14.0f + 18.0f}}}), sum);
    }

    @Test
    void reduceSum_test_5x3x4_dim_minusOne_noKeepDims() {
        ITensor ndarray = sciCore.ndarray(new float[][][]{
                {{10.0f, 12.0f, 14.0f, 16.0f}, {12.0f, 14.0f, 16.0f, 18.0f}, {14.0f, 16.0f, 18.0f, 20.0f}},
                {{16.0f, 18.0f, 20.0f, 22.0f}, {18.0f, 20.0f, 22.0f, 24.0f}, {20.0f, 22.0f, 24.0f, 26.0f}},
                {{22.0f, 24.0f, 26.0f, 28.0f}, {24.0f, 26.0f, 28.0f, 30.0f}, {26.0f, 28.0f, 30.0f, 32.0f}},
                {{28.0f, 30.0f, 32.0f, 34.0f}, {30.0f, 32.0f, 34.0f, 36.0f}, {32.0f, 34.0f, 36.0f, 38.0f}},
                {{34.0f, 36.0f, 38.0f, 40.0f}, {36.0f, 38.0f, 40.0f, 42.0f}, {38.0f, 40.0f, 42.0f, 44.0f}}});
        ITensor sum = ndarray.reduceSum(-1);
        assertEquals(sciCore.scalar(10.0f + 12.0f + 14.0f + 16.0f + 12.0f + 14.0f + 16.0f + 18.0f + 14.0f + 16.0f + 18.0f + 20.0f
                + 16.0f + 18.0f + 20.0f + 22.0f + 18.0f + 20.0f + 22.0f + 24.0f + 20.0f + 22.0f + 24.0f + 26.0f
                + 22.0f + 24.0f + 26.0f + 28.0f + 24.0f + 26.0f + 28.0f + 30.0f + 26.0f + 28.0f + 30.0f + 32.0f
                + 28.0f + 30.0f + 32.0f + 34.0f + 30.0f + 32.0f + 34.0f + 36.0f + 32.0f + 34.0f + 36.0f + 38.0f
                + 34.0f + 36.0f + 38.0f + 40.0f + 36.0f + 38.0f + 40.0f + 42.0f + 38.0f + 40.0f + 42.0f + 44.0f), sum);
    }

    @Test
    void reduceSum_test_5x3x4_dim_minusOne_keepDims() {
        ITensor ndarray = sciCore.ndarray(new float[][][]{
                {{10.0f, 12.0f, 14.0f, 16.0f}, {12.0f, 14.0f, 16.0f, 18.0f}, {14.0f, 16.0f, 18.0f, 20.0f}},
                {{16.0f, 18.0f, 20.0f, 22.0f}, {18.0f, 20.0f, 22.0f, 24.0f}, {20.0f, 22.0f, 24.0f, 26.0f}},
                {{22.0f, 24.0f, 26.0f, 28.0f}, {24.0f, 26.0f, 28.0f, 30.0f}, {26.0f, 28.0f, 30.0f, 32.0f}},
                {{28.0f, 30.0f, 32.0f, 34.0f}, {30.0f, 32.0f, 34.0f, 36.0f}, {32.0f, 34.0f, 36.0f, 38.0f}},
                {{34.0f, 36.0f, 38.0f, 40.0f}, {36.0f, 38.0f, 40.0f, 42.0f}, {38.0f, 40.0f, 42.0f, 44.0f}}});
        ITensor sum = ndarray.reduceSum(-1, true);
        assertEquals(sciCore.ndarray(new float[][][]{{{
                10.0f + 12.0f + 14.0f + 16.0f + 12.0f + 14.0f + 16.0f + 18.0f + 14.0f + 16.0f + 18.0f + 20.0f +
                        16.0f + 18.0f + 20.0f + 22.0f + 18.0f + 20.0f + 22.0f + 24.0f + 20.0f + 22.0f + 24.0f + 26.0f +
                        22.0f + 24.0f + 26.0f + 28.0f + 24.0f + 26.0f + 28.0f + 30.0f + 26.0f + 28.0f + 30.0f + 32.0f +
                        28.0f + 30.0f + 32.0f + 34.0f + 30.0f + 32.0f + 34.0f + 36.0f + 32.0f + 34.0f + 36.0f + 38.0f +
                        34.0f + 36.0f + 38.0f + 40.0f + 36.0f + 38.0f + 40.0f + 42.0f + 38.0f + 40.0f + 42.0f + 44.0f}}}), sum);
    }

    @Test
    void reduceSum_test_5x3x4_dim0_noKeepDims() {
        ITensor ndarray = sciCore.ndarray(new float[][][]{
                {{10.0f, 12.0f, 14.0f, 16.0f}, {12.0f, 14.0f, 16.0f, 18.0f}, {14.0f, 16.0f, 18.0f, 20.0f}},
                {{16.0f, 18.0f, 20.0f, 22.0f}, {18.0f, 20.0f, 22.0f, 24.0f}, {20.0f, 22.0f, 24.0f, 26.0f}},
                {{22.0f, 24.0f, 26.0f, 28.0f}, {24.0f, 26.0f, 28.0f, 30.0f}, {26.0f, 28.0f, 30.0f, 32.0f}},
                {{28.0f, 30.0f, 32.0f, 34.0f}, {30.0f, 32.0f, 34.0f, 36.0f}, {32.0f, 34.0f, 36.0f, 38.0f}},
                {{34.0f, 36.0f, 38.0f, 40.0f}, {36.0f, 38.0f, 40.0f, 42.0f}, {38.0f, 40.0f, 42.0f, 44.0f}}});
        ITensor sum = ndarray.reduceSum(0);
        assertArrayEquals(new long[]{3, 4}, sum.getShape());
        assertEquals(sciCore.matrix(new float[][]{
                {10.0f + 16.0f + 22.0f + 28.0f + 34.0f, 12.0f + 18.0f + 24.0f + 30.0f + 36.0f, 14.0f + 20.0f + 26.0f + 32.0f + 38.0f, 16.0f + 22.0f + 28.0f + 34.0f + 40.0f},
                {12.0f + 18.0f + 24.0f + 30.0f + 36.0f, 14.0f + 20.0f + 26.0f + 32.0f + 38.0f, 16.0f + 22.0f + 28.0f + 34.0f + 40.0f, 18.0f + 24.0f + 30.0f + 36.0f + 42.0f},
                {14.0f + 20.0f + 26.0f + 32.0f + 38.0f, 16.0f + 22.0f + 28.0f + 34.0f + 40.0f, 18.0f + 24.0f + 30.0f + 36.0f + 42.0f, 20.0f + 26.0f + 32.0f + 38.0f + 44.0f}}), sum);
    }

    @Test
    void reduceSum_test_5x3x4_dim0_keepDims() {
        ITensor ndarray = sciCore.ndarray(new float[][][]{
                {{10.0f, 12.0f, 14.0f, 16.0f}, {12.0f, 14.0f, 16.0f, 18.0f}, {14.0f, 16.0f, 18.0f, 20.0f}},
                {{16.0f, 18.0f, 20.0f, 22.0f}, {18.0f, 20.0f, 22.0f, 24.0f}, {20.0f, 22.0f, 24.0f, 26.0f}},
                {{22.0f, 24.0f, 26.0f, 28.0f}, {24.0f, 26.0f, 28.0f, 30.0f}, {26.0f, 28.0f, 30.0f, 32.0f}},
                {{28.0f, 30.0f, 32.0f, 34.0f}, {30.0f, 32.0f, 34.0f, 36.0f}, {32.0f, 34.0f, 36.0f, 38.0f}},
                {{34.0f, 36.0f, 38.0f, 40.0f}, {36.0f, 38.0f, 40.0f, 42.0f}, {38.0f, 40.0f, 42.0f, 44.0f}}});
        ITensor sum = ndarray.reduceSum(0, true);
        assertEquals(sciCore.ndarray(new float[][][]{
                {{10.0f + 16.0f + 22.0f + 28.0f + 34.0f, 12.0f + 18.0f + 24.0f + 30.0f + 36.0f, 14.0f + 20.0f + 26.0f + 32.0f + 38.0f, 16.0f + 22.0f + 28.0f + 34.0f + 40.0f},
                        {12.0f + 18.0f + 24.0f + 30.0f + 36.0f, 14.0f + 20.0f + 26.0f + 32.0f + 38.0f, 16.0f + 22.0f + 28.0f + 34.0f + 40.0f, 18.0f + 24.0f + 30.0f + 36.0f + 42.0f},
                        {14.0f + 20.0f + 26.0f + 32.0f + 38.0f, 16.0f + 22.0f + 28.0f + 34.0f + 40.0f, 18.0f + 24.0f + 30.0f + 36.0f + 42.0f, 20.0f + 26.0f + 32.0f + 38.0f + 44.0f}}}), sum);
    }

    @Test
    void reduceSum_test_5x3x4_dim1_noKeepDims() {
        ITensor ndarray = sciCore.ndarray(new float[][][]{
                {{10.0f, 12.0f, 14.0f, 16.0f}, {12.0f, 14.0f, 16.0f, 18.0f}, {14.0f, 16.0f, 18.0f, 20.0f}},
                {{16.0f, 18.0f, 20.0f, 22.0f}, {18.0f, 20.0f, 22.0f, 24.0f}, {20.0f, 22.0f, 24.0f, 26.0f}},
                {{22.0f, 24.0f, 26.0f, 28.0f}, {24.0f, 26.0f, 28.0f, 30.0f}, {26.0f, 28.0f, 30.0f, 32.0f}},
                {{28.0f, 30.0f, 32.0f, 34.0f}, {30.0f, 32.0f, 34.0f, 36.0f}, {32.0f, 34.0f, 36.0f, 38.0f}},
                {{34.0f, 36.0f, 38.0f, 40.0f}, {36.0f, 38.0f, 40.0f, 42.0f}, {38.0f, 40.0f, 42.0f, 44.0f}}});
        ITensor sum = ndarray.reduceSum(1);
        assertEquals(sciCore.matrix(new float[][]{
                {10.0f + 12.0f + 14.0f, 12.0f + 14.0f + 16.0f, 14.0f + 16.0f + 18.0f, 16.0f + 18.0f + 20.0f},
                {16.0f + 18.0f + 20.0f, 18.0f + 20.0f + 22.0f, 20.0f + 22.0f + 24.0f, 22.0f + 24.0f + 26.0f},
                {22.0f + 24.0f + 26.0f, 24.0f + 26.0f + 28.0f, 26.0f + 28.0f + 30.0f, 28.0f + 30.0f + 32.0f},
                {28.0f + 30.0f + 32.0f, 30.0f + 32.0f + 34.0f, 32.0f + 34.0f + 36.0f, 34.0f + 36.0f + 38.0f},
                {34.0f + 36.0f + 38.0f, 36.0f + 38.0f + 40.0f, 38.0f + 40.0f + 42.0f, 40.0f + 42.0f + 44.0f}}), sum);
    }

    @Test
    void reduceSum_test_5x3x4_dim1_keepDims() {
        ITensor ndarray = sciCore.ndarray(new float[][][]{
                {{10.0f, 12.0f, 14.0f, 16.0f}, {12.0f, 14.0f, 16.0f, 18.0f}, {14.0f, 16.0f, 18.0f, 20.0f}},
                {{16.0f, 18.0f, 20.0f, 22.0f}, {18.0f, 20.0f, 22.0f, 24.0f}, {20.0f, 22.0f, 24.0f, 26.0f}},
                {{22.0f, 24.0f, 26.0f, 28.0f}, {24.0f, 26.0f, 28.0f, 30.0f}, {26.0f, 28.0f, 30.0f, 32.0f}},
                {{28.0f, 30.0f, 32.0f, 34.0f}, {30.0f, 32.0f, 34.0f, 36.0f}, {32.0f, 34.0f, 36.0f, 38.0f}},
                {{34.0f, 36.0f, 38.0f, 40.0f}, {36.0f, 38.0f, 40.0f, 42.0f}, {38.0f, 40.0f, 42.0f, 44.0f}}});
        ITensor sum = ndarray.reduceSum(1, true);
        assertEquals(sciCore.ndarray(new float[][][]{
                {{10.0f + 12.0f + 14.0f, 12.0f + 14.0f + 16.0f, 14.0f + 16.0f + 18.0f, 16.0f + 18.0f + 20.0f}},
                {{16.0f + 18.0f + 20.0f, 18.0f + 20.0f + 22.0f, 20.0f + 22.0f + 24.0f, 22.0f + 24.0f + 26.0f}},
                {{22.0f + 24.0f + 26.0f, 24.0f + 26.0f + 28.0f, 26.0f + 28.0f + 30.0f, 28.0f + 30.0f + 32.0f}},
                {{28.0f + 30.0f + 32.0f, 30.0f + 32.0f + 34.0f, 32.0f + 34.0f + 36.0f, 34.0f + 36.0f + 38.0f}},
                {{34.0f + 36.0f + 38.0f, 36.0f + 38.0f + 40.0f, 38.0f + 40.0f + 42.0f, 40.0f + 42.0f + 44.0f}}}), sum);
    }

    @Test
    void reduceSum_test_5x3x4_dim2_noKeepDims() {
        ITensor ndarray = sciCore.ndarray(new float[][][]{
                {{10.0f, 12.0f, 14.0f, 16.0f}, {12.0f, 14.0f, 16.0f, 18.0f}, {14.0f, 16.0f, 18.0f, 20.0f}},
                {{16.0f, 18.0f, 20.0f, 22.0f}, {18.0f, 20.0f, 22.0f, 24.0f}, {20.0f, 22.0f, 24.0f, 26.0f}},
                {{22.0f, 24.0f, 26.0f, 28.0f}, {24.0f, 26.0f, 28.0f, 30.0f}, {26.0f, 28.0f, 30.0f, 32.0f}},
                {{28.0f, 30.0f, 32.0f, 34.0f}, {30.0f, 32.0f, 34.0f, 36.0f}, {32.0f, 34.0f, 36.0f, 38.0f}},
                {{34.0f, 36.0f, 38.0f, 40.0f}, {36.0f, 38.0f, 40.0f, 42.0f}, {38.0f, 40.0f, 42.0f, 44.0f}}});
        ITensor sum = ndarray.reduceSum(2);
        assertArrayEquals(new long[]{5, 3}, sum.getShape());
        assertEquals(sciCore.matrix(new float[][]{
                {10 + 12 + 14 + 16, 12 + 14 + 16 + 18, 14 + 16 + 18 + 20},
                {16 + 18 + 20 + 22, 18 + 20 + 22 + 24, 20 + 22 + 24 + 26},
                {22 + 24 + 26 + 28, 24 + 26 + 28 + 30, 26 + 28 + 30 + 32},
                {28 + 30 + 32 + 34, 30 + 32 + 34 + 36, 32 + 34 + 36 + 38},
                {34 + 36 + 38 + 40, 36 + 38 + 40 + 42, 38 + 40 + 42 + 44}}), sum);
    }

    @Test
    void reduceSum_test_5x3x4_dim2_keepDims() {
        ITensor ndarray = sciCore.ndarray(new float[][][]{
                {{10.0f, 12.0f, 14.0f, 16.0f}, {12.0f, 14.0f, 16.0f, 18.0f}, {14.0f, 16.0f, 18.0f, 20.0f}},
                {{16.0f, 18.0f, 20.0f, 22.0f}, {18.0f, 20.0f, 22.0f, 24.0f}, {20.0f, 22.0f, 24.0f, 26.0f}},
                {{22.0f, 24.0f, 26.0f, 28.0f}, {24.0f, 26.0f, 28.0f, 30.0f}, {26.0f, 28.0f, 30.0f, 32.0f}},
                {{28.0f, 30.0f, 32.0f, 34.0f}, {30.0f, 32.0f, 34.0f, 36.0f}, {32.0f, 34.0f, 36.0f, 38.0f}},
                {{34.0f, 36.0f, 38.0f, 40.0f}, {36.0f, 38.0f, 40.0f, 42.0f}, {38.0f, 40.0f, 42.0f, 44.0f}}});
        ITensor sum = ndarray.reduceSum(2, true);
        assertArrayEquals(new long[]{5, 3, 1}, sum.getShape());
        assertEquals(sciCore.ndarray(new float[][][]{
                {{10 + 12 + 14 + 16}, {12 + 14 + 16 + 18}, {14 + 16 + 18 + 20}},
                {{16 + 18 + 20 + 22}, {18 + 20 + 22 + 24}, {20 + 22 + 24 + 26}},
                {{22 + 24 + 26 + 28}, {24 + 26 + 28 + 30}, {26 + 28 + 30 + 32}},
                {{28 + 30 + 32 + 34}, {30 + 32 + 34 + 36}, {32 + 34 + 36 + 38}},
                {{34 + 36 + 38 + 40}, {36 + 38 + 40 + 42}, {38 + 40 + 42 + 44}}}), sum);
    }

    @Test
    void plus_test_1x1_plus_1x1() {
        ITensor a = sciCore.matrix(new float[][]{{1.0f}});
        ITensor b = sciCore.matrix(new float[][]{{2.0f}});
        ITensor result = a.plus(b);
        assertEquals(3.0f, result.getFloat(0, 0), EPSILON);
    }

    @Test
    void plus_test_3x2_plus_1x1() {
        ITensor a = sciCore.matrix(new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
        ITensor b = sciCore.matrix(new float[][]{{2.0f}});
        ITensor result = a.plus(b);
        assertEquals(sciCore.matrix(new float[][]{{3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}}), result);
    }

    @Test
    void plus_test_5x3x2_plus_3x2_2dBroadcast() {
        // (5, 3, 2) + (3, 2) = (5, 3, 2)
        ITensor a = sciCore.ndarray(new float[][][]{
                {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
                {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}},
                {{13.0f, 14.0f}, {15.0f, 16.0f}, {17.0f, 18.0f}},
                {{19.0f, 20.0f}, {21.0f, 22.0f}, {23.0f, 24.0f}},
                {{25.0f, 26.0f}, {27.0f, 28.0f}, {29.0f, 30.0f}}
        });
        ITensor b = sciCore.ndarray(new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
        ITensor c = a.plus(b);
        assertEquals(sciCore.ndarray(new float[][][]{
                {{2.0f, 4.0f}, {6.0f, 8.0f}, {10.0f, 12.0f}},
                {{8.0f, 10.0f}, {12.0f, 14.0f}, {16.0f, 18.0f}},
                {{14.0f, 16.0f}, {18.0f, 20.0f}, {22.0f, 24.0f}},
                {{20.0f, 22.0f}, {24.0f, 26.0f}, {28.0f, 30.0f}},
                {{26.0f, 28.0f}, {30.0f, 32.0f}, {34.0f, 36.0f}}}), c);
    }

    @Test
    void plus_test_2x3x4_plus_3x3x4_broadcastImpossible_failure() {
        // (2, 3, 4) + (3, 3, 4)
        ITensor a = sciCore.ndarray(new float[][][]{
                {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}},
                {{13.0f, 14.0f, 15.0f, 16.0f}, {17.0f, 18.0f, 19.0f, 20.0f}, {21.0f, 22.0f, 23.0f, 24.0f}}});
        ITensor b = sciCore.ndarray(new float[][][]{
                {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}},
                {{13.0f, 14.0f, 15.0f, 16.0f}, {17.0f, 18.0f, 19.0f, 20.0f}, {21.0f, 22.0f, 23.0f, 24.0f}},
                {{25.0f, 26.0f, 27.0f, 28.0f}, {29.0f, 30.0f, 31.0f, 32.0f}, {33.0f, 34.0f, 35.0f, 36.0f}}});
        assertThrows(IllegalArgumentException.class, () -> a.plus(b));
    }

    @Test
    void plus_test_3d_plus_3d_broadcast_firstDimIsOne() {
        // (2, 3, 2) + (1, 3, 2) = (2, 3, 2)
        ITensor a = sciCore.ndarray(new float[][][]{
                {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
                {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}},
        });
        ITensor b = sciCore.ndarray(new float[][][]{
                {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
        });
        ITensor c = a.plus(b);
        assertEquals(sciCore.ndarray(new float[][][]{
                {{2.0f, 4.0f}, {6.0f, 8.0f}, {10.0f, 12.0f}},
                {{8.0f, 10.0f}, {12.0f, 14.0f}, {16.0f, 18.0f}}}), c);
    }

    @Test
    void minus_test_1x1_minus_1x1() {
        ITensor a = sciCore.matrix(new float[][]{{1.0f}});
        ITensor b = sciCore.matrix(new float[][]{{2.0f}});
        ITensor result = a.minus(b);
        assertEquals(-1.0f, result.getFloat(0, 0), EPSILON);
    }

    @Test
    void minus_test_2x2_Minus_2x2() {
        ITensor a = sciCore.matrix(new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}});
        ITensor b = sciCore.matrix(new float[][]{{2.0f, 3.0f}, {4.0f, 5.0f}});
        ITensor result = a.minus(b);
        assertEquals(sciCore.matrix(new float[][]{{-1.0f, -1.0f}, {-1.0f, -1.0f}}), result);
    }

    @Test
    void minus_test_5x3x2_minus_3x2_2dBroadcast() {
        // (5, 3, 2) - (3, 2) = (5, 3, 2)
        ITensor a = sciCore.ndarray(new float[][][]{
                {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
                {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}},
                {{13.0f, 14.0f}, {15.0f, 16.0f}, {17.0f, 18.0f}},
                {{19.0f, 20.0f}, {21.0f, 22.0f}, {23.0f, 24.0f}},
                {{25.0f, 26.0f}, {27.0f, 28.0f}, {29.0f, 30.0f}}
        });
        ITensor b = sciCore.ndarray(new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
        ITensor c = a.minus(b);
        assertEquals(sciCore.ndarray(new float[][][]{
                {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}},
                {{6.0f, 6.0f}, {6.0f, 6.0f}, {6.0f, 6.0f}},
                {{12.0f, 12.0f}, {12.0f, 12.0f}, {12.0f, 12.0f}},
                {{18.0f, 18.0f}, {18.0f, 18.0f}, {18.0f, 18.0f}},
                {{24.0f, 24.0f}, {24.0f, 24.0f}, {24.0f, 24.0f}}}), c);
    }

    @Test
    void minus_test_2x3x4_minus_3x3x4_broadcastImpossible_failure() {
        // (2, 3, 4) - (3, 3, 4)
        ITensor a = sciCore.ndarray(new float[][][]{
                {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}},
                {{13.0f, 14.0f, 15.0f, 16.0f}, {17.0f, 18.0f, 19.0f, 20.0f}, {21.0f, 22.0f, 23.0f, 24.0f}}});
        ITensor b = sciCore.ndarray(new float[][][]{
                {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}},
                {{13.0f, 14.0f, 15.0f, 16.0f}, {17.0f, 18.0f, 19.0f, 20.0f}, {21.0f, 22.0f, 23.0f, 24.0f}},
                {{25.0f, 26.0f, 27.0f, 28.0f}, {29.0f, 30.0f, 31.0f, 32.0f}, {33.0f, 34.0f, 35.0f, 36.0f}}});
        assertThrows(IllegalArgumentException.class, () -> a.minus(b));
    }

    @Test
    void minus_test_3d_minus_3d_broadcast_firstDimIsOne() {
        // (2, 3, 2) - (1, 3, 2) = (2, 3, 2)
        ITensor a = sciCore.ndarray(new float[][][]{
                {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
                {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}},
        });
        ITensor b = sciCore.ndarray(new float[][][]{
                {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
        });
        ITensor c = a.minus(b);
        assertEquals(sciCore.ndarray(new float[][][]{
                {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}},
                {{6.0f, 6.0f}, {6.0f, 6.0f}, {6.0f, 6.0f}}}), c);
    }

    @Test
    void multiply_test_tensorByTensorElementWise_success() {
        ITensor a = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}});
        ITensor b = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}});
        ITensor c = a.multiply(b);

        assertEquals(sciCore.matrix(new float[][]{{1.0f, 4.0f, 9.0f}, {16.0f, 25.0f, 36.0f}, {49.0f, 64.0f, 81.0f}}), c);
    }

    @Test
    void multiply_test_tensorByTensorElementWise_differentShape_failure() {
        ITensor a = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}});
        ITensor b = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
        assertThrows(IllegalArgumentException.class, () -> a.multiply(b));
    }

    @Test
    void multiply_test_tensorByScalar_success() {
        ITensor a = sciCore.matrix(new float[][]{{2, 3}});
        ITensor b = sciCore.scalar(10);

        ITensor c = a.multiply(b);
        assertEquals(20, c.getFloat(0, 0), EPSILON);
        assertEquals(30, c.getFloat(0, 1), EPSILON);
    }

    @Test
    void multiply_test_tensorByTensorDimensionWiseSingleDim_success() {
        ITensor a = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}});
        ITensor b = sciCore.matrix(new float[][]{{4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}});
        ITensor c = a.multiply(b);

        assertEquals(sciCore.matrix(new float[][]{{4.0f, 10.0f, 18.0f}, {7.0f, 16.0f, 27.0f}}), c);
    }

    @Test
    void multiply_test_tensorByTensorDimensionWiseMultipleDim_success() {
        ITensor a = sciCore.matrix(new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}});
        ITensor b = sciCore.ndarray(new float[][][]{{{4.0f, 5.0f}, {6.0f, 7.0f}}, {{8.0f, 9.0f}, {10.0f, 11.0f}}});

        ITensor c = a.multiply(b);
        assertEquals(sciCore.ndarray(new float[][][]{{
                {4.0f, 10.0f}, {18.0f, 28.0f}},
                {{8.0f, 18.0f}, {30.0f, 44.0f}}
        }), c);
    }
}