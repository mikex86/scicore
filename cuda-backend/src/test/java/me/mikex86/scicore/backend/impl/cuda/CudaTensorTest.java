package me.mikex86.scicore.backend.impl.cuda;

import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.SciCore;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertThrows;


@TestInstance(TestInstance.Lifecycle.PER_METHOD)
public class CudaTensorTest {

    ISciCore sciCore;

    @BeforeEach
    void setUp() {
        sciCore = new SciCore();
        sciCore.setBackend(SciCore.BackendType.CUDA);
    }


    @Test
    void testCreateByteArray() {
        byte[] array = new byte[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        ITensor tensor = sciCore.array(array);
        for (int i = 0; i < array.length; i++) {
            assertEquals(array[i], tensor.getByte(i));
        }
    }

    @Test
    void testCreateShortArray() {
        short[] array = new short[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        ITensor tensor = sciCore.array(array);
        for (int i = 0; i < array.length; i++) {
            assertEquals(array[i], tensor.getShort(i));
        }
    }

    @Test
    void testCreateIntArray() {
        int[] array = new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        ITensor tensor = sciCore.array(array);
        for (int i = 0; i < array.length; i++) {
            assertEquals(array[i], tensor.getInt(i));
        }
    }

    @Test
    void testCreateLongArray() {
        long[] array = new long[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        ITensor tensor = sciCore.array(array);
        for (int i = 0; i < array.length; i++) {
            assertEquals(array[i], tensor.getLong(i));
        }
    }

    @Test
    void testCreateFloatArray() {
        float[] array = new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        ITensor tensor = sciCore.array(array);
        for (int i = 0; i < array.length; i++) {
            assertEquals(array[i], tensor.getFloat(i));
        }
    }

    @Test
    void testCreateDoubleArray() {
        double[] array = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        ITensor tensor = sciCore.array(array);
        for (int i = 0; i < array.length; i++) {
            assertEquals(array[i], tensor.getDouble(i));
        }
    }

    @Test
    void testCreateBooleanArray() {
        boolean[] array = new boolean[]{true, false, true, false, true, false, true, false, false, false};
        ITensor tensor = sciCore.array(array);
        for (int i = 0; i < array.length; i++) {
            assertEquals(array[i], tensor.getBoolean(i));
        }
    }

    @Test
    void testSetByte() {
        ITensor tensor = sciCore.array(new byte[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        assertEquals(5, tensor.getByte(4));
        tensor.setByte((byte) 42, 4);
        assertEquals(42, tensor.getByte(4));
    }

    @Test
    void testSetShort() {
        ITensor tensor = sciCore.array(new short[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        assertEquals(5, tensor.getShort(4));
        tensor.setShort((short) 42, 4);
        assertEquals(42, tensor.getShort(4));
    }

    @Test
    void testSetInt() {
        ITensor tensor = sciCore.array(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        assertEquals(5, tensor.getInt(4));
        tensor.setInt(42, 4);
        assertEquals(42, tensor.getInt(4));
    }

    @Test
    void testSetLong() {
        ITensor tensor = sciCore.array(new long[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        assertEquals(5, tensor.getLong(4));
        tensor.setLong(42, 4);
        assertEquals(42, tensor.getLong(4));
    }

    @Test
    void testSetFloat() {
        ITensor tensor = sciCore.array(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        assertEquals(5, tensor.getFloat(4));
        tensor.setFloat(42, 4);
        assertEquals(42, tensor.getFloat(4));
    }

    @Test
    void testSetDouble() {
        ITensor tensor = sciCore.array(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        assertEquals(5, tensor.getDouble(4));
        tensor.setDouble(42, 4);
        assertEquals(42, tensor.getDouble(4));
    }

    @Test
    void testCreateByteMatrix() {
        byte[][] data = new byte[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        };
        ITensor matrix = sciCore.matrix(data);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(data[i][j], matrix.getByte(i, j));
            }
        }
    }

    @Test
    void testCreateShortMatrix() {
        short[][] data = new short[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        };
        ITensor matrix = sciCore.matrix(data);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(data[i][j], matrix.getShort(i, j));
            }
        }
    }

    @Test
    void testCreateIntMatrix() {
        int[][] data = new int[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        };
        ITensor matrix = sciCore.matrix(data);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(data[i][j], matrix.getInt(i, j));
            }
        }
    }

    @Test
    void testCreateLongMatrix() {
        long[][] data = new long[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        };
        ITensor matrix = sciCore.matrix(data);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(data[i][j], matrix.getLong(i, j));
            }
        }
    }

    @Test
    void testCreateFloatMatrix() {
        float[][] data = new float[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        };
        ITensor matrix = sciCore.matrix(data);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(data[i][j], matrix.getFloat(i, j));
            }
        }
    }

    @Test
    void testCreateDoubleMatrix() {
        double[][] data = new double[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        };
        ITensor matrix = sciCore.matrix(data);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(data[i][j], matrix.getDouble(i, j));
            }
        }
    }

    @Test
    void testCreateBooleanMatrix() {
        boolean[][] data = new boolean[][]{
                {true, false, true},
                {false, true, false},
                {true, false, false}
        };
        ITensor matrix = sciCore.matrix(data);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(data[i][j], matrix.getBoolean(i, j));
            }
        }
    }

    @Test
    void testCreateByteMatrixAndSetByte() {
        byte[][] data = new byte[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        };
        ITensor matrix = sciCore.matrix(data);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(data[i][j], matrix.getByte(i, j));
            }
        }
        matrix.setByte((byte) 42, 1, 1);
        assertEquals(42, matrix.getByte(1, 1));
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i == 1 && j == 1) {
                    continue;
                }
                assertEquals(data[i][j], matrix.getByte(i, j));
            }
        }
    }

    @Test
    void testCreateShortMatrixAndSetShort() {
        short[][] data = new short[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        };
        ITensor matrix = sciCore.matrix(data);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(data[i][j], matrix.getShort(i, j));
            }
        }
        matrix.setShort((short) 42, 1, 1);
        assertEquals(42, matrix.getShort(1, 1));
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i == 1 && j == 1) {
                    continue;
                }
                assertEquals(data[i][j], matrix.getShort(i, j));
            }
        }
    }

    @Test
    void testCreateIntMatrixAndSetInt() {
        int[][] data = new int[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        };
        ITensor matrix = sciCore.matrix(data);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(data[i][j], matrix.getInt(i, j));
            }
        }
        matrix.setInt(42, 1, 1);
        assertEquals(42, matrix.getInt(1, 1));
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i == 1 && j == 1) {
                    continue;
                }
                assertEquals(data[i][j], matrix.getInt(i, j));
            }
        }
    }

    @Test
    void testCreateLongMatrixAndSetLong() {
        long[][] data = new long[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        };
        ITensor matrix = sciCore.matrix(data);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(data[i][j], matrix.getLong(i, j));
            }
        }
        matrix.setLong(42, 1, 1);
        assertEquals(42, matrix.getLong(1, 1));
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i == 1 && j == 1) {
                    continue;
                }
                assertEquals(data[i][j], matrix.getLong(i, j));
            }
        }
    }

    @Test
    void testCreateFloatMatrixAndSetFloat() {
        float[][] data = new float[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        };
        ITensor matrix = sciCore.matrix(data);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(data[i][j], matrix.getFloat(i, j));
            }
        }
        matrix.setFloat(42, 1, 1);
        assertEquals(42, matrix.getFloat(1, 1));
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i == 1 && j == 1) {
                    continue;
                }
                assertEquals(data[i][j], matrix.getFloat(i, j));
            }
        }
    }

    @Test
    void testCreateDoubleMatrixAndSetDouble() {
        double[][] data = new double[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        };
        ITensor matrix = sciCore.matrix(data);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(data[i][j], matrix.getDouble(i, j));
            }
        }
        matrix.setDouble(42, 1, 1);
        assertEquals(42, matrix.getDouble(1, 1));
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i == 1 && j == 1) {
                    continue;
                }
                assertEquals(data[i][j], matrix.getDouble(i, j));
            }
        }
    }

    @Test
    void testCreateBooleanMatrixAndSetBoolean() {
        boolean[][] data = new boolean[][]{
                {true, false, true},
                {false, true, false},
                {true, false, true}
        };
        ITensor matrix = sciCore.matrix(data);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(data[i][j], matrix.getBoolean(i, j));
            }
        }
        matrix.setBoolean(false, 1, 1);
        assertFalse(matrix.getBoolean(1, 1));
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i == 1 && j == 1) {
                    continue;
                }
                assertEquals(data[i][j], matrix.getBoolean(i, j));
            }
        }
    }

    @Test
    void testCreateByteNdArray() {
        byte[][][] data = new byte[][][]{
                {
                        {1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9}
                },
                {
                        {10, 11, 12},
                        {13, 14, 15},
                        {16, 17, 18}
                }
        };
        ITensor ndarray = sciCore.ndarray(data);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    assertEquals(data[i][j][k], ndarray.getByte(i, j, k));
                }
            }
        }
    }

    @Test
    void testCreateShortNdArray() {
        short[][][] data = new short[][][]{
                {
                        {1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9}
                },
                {
                        {10, 11, 12},
                        {13, 14, 15},
                        {16, 17, 18}
                }
        };
        ITensor ndarray = sciCore.ndarray(data);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    assertEquals(data[i][j][k], ndarray.getShort(i, j, k));
                }
            }
        }
    }

    @Test
    void testCreateIntNdArray() {
        int[][][] data = new int[][][]{
                {
                        {1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9}
                },
                {
                        {10, 11, 12},
                        {13, 14, 15},
                        {16, 17, 18}
                }
        };
        ITensor ndarray = sciCore.ndarray(data);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    assertEquals(data[i][j][k], ndarray.getInt(i, j, k));
                }
            }
        }
    }

    @Test
    void testCreateLongNdArray() {
        long[][][] data = new long[][][]{
                {
                        {1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9}
                },
                {
                        {10, 11, 12},
                        {13, 14, 15},
                        {16, 17, 18}
                }
        };
        ITensor ndarray = sciCore.ndarray(data);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    assertEquals(data[i][j][k], ndarray.getLong(i, j, k));
                }
            }
        }
    }

    @Test
    void testCreateFloatNdArray() {
        float[][][] data = new float[][][]{
                {
                        {1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9}
                },
                {
                        {10, 11, 12},
                        {13, 14, 15},
                        {16, 17, 18}
                }
        };
        ITensor ndarray = sciCore.ndarray(data);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    assertEquals(data[i][j][k], ndarray.getFloat(i, j, k));
                }
            }
        }
    }

    @Test
    void testCreateDoubleNdArray() {
        double[][][] data = new double[][][]{
                {
                        {1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9}
                },
                {
                        {10, 11, 12},
                        {13, 14, 15},
                        {16, 17, 18}
                }
        };
        ITensor ndarray = sciCore.ndarray(data);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    assertEquals(data[i][j][k], ndarray.getDouble(i, j, k));
                }
            }
        }
    }

    @Test
    void testCreateBooleanNdArray() {
        boolean[][][] data = new boolean[][][]{
                {
                        {true, false, true},
                        {false, true, false},
                        {true, false, true}
                },
                {
                        {false, true, false},
                        {true, false, true},
                        {false, true, false}
                }
        };
        ITensor ndarray = sciCore.ndarray(data);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    assertEquals(data[i][j][k], ndarray.getBoolean(i, j, k));
                }
            }
        }
    }

    @Test
    void testCreateByteNdArrayAndSetByte() {
        byte[][][] data = new byte[][][]{
                {
                        {1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9}
                },
                {
                        {10, 11, 12},
                        {13, 14, 15},
                        {16, 17, 18}
                }
        };
        ITensor ndarray = sciCore.ndarray(data);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    assertEquals(data[i][j][k], ndarray.getByte(i, j, k));
                }
            }
        }
        ndarray.setByte((byte) 0, 1, 2, 2);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    if (i == 1 && j == 2 && k == 2) {
                        assertEquals(0, ndarray.getByte(i, j, k));
                    } else {
                        assertEquals(data[i][j][k], ndarray.getByte(i, j, k));
                    }
                }
            }
        }
    }

    @Test
    void testCreateShortNdArrayAndSetShort() {
        short[][][] data = new short[][][]{
                {
                        {1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9}
                },
                {
                        {10, 11, 12},
                        {13, 14, 15},
                        {16, 17, 18}
                }
        };
        ITensor ndarray = sciCore.ndarray(data);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    assertEquals(data[i][j][k], ndarray.getShort(i, j, k));
                }
            }
        }
        ndarray.setShort((short) 0, 1, 2, 2);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    if (i == 1 && j == 2 && k == 2) {
                        assertEquals(0, ndarray.getShort(i, j, k));
                    } else {
                        assertEquals(data[i][j][k], ndarray.getShort(i, j, k));
                    }
                }
            }
        }
    }

    @Test
    void testCreateIntNdArrayAndSetInt() {
        int[][][] data = new int[][][]{
                {
                        {1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9}
                },
                {
                        {10, 11, 12},
                        {13, 14, 15},
                        {16, 17, 18}
                }
        };
        ITensor ndarray = sciCore.ndarray(data);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    assertEquals(data[i][j][k], ndarray.getInt(i, j, k));
                }
            }
        }
        ndarray.setInt(0, 1, 2, 2);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    if (i == 1 && j == 2 && k == 2) {
                        assertEquals(0, ndarray.getInt(i, j, k));
                    } else {
                        assertEquals(data[i][j][k], ndarray.getInt(i, j, k));
                    }
                }
            }
        }
    }

    @Test
    void testCreateLongNdArrayAndSetLong() {
        long[][][] data = new long[][][]{
                {
                        {1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9}
                },
                {
                        {10, 11, 12},
                        {13, 14, 15},
                        {16, 17, 18}
                }
        };
        ITensor ndarray = sciCore.ndarray(data);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    assertEquals(data[i][j][k], ndarray.getLong(i, j, k));
                }
            }
        }
        ndarray.setLong(0, 1, 2, 2);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    if (i == 1 && j == 2 && k == 2) {
                        assertEquals(0, ndarray.getLong(i, j, k));
                    } else {
                        assertEquals(data[i][j][k], ndarray.getLong(i, j, k));
                    }
                }
            }
        }
    }

    @Test
    void testCreateFloatNdArrayAndSetFloat() {
        float[][][] data = new float[][][]{
                {
                        {1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9}
                },
                {
                        {10, 11, 12},
                        {13, 14, 15},
                        {16, 17, 18}
                }
        };
        ITensor ndarray = sciCore.ndarray(data);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    assertEquals(data[i][j][k], ndarray.getFloat(i, j, k));
                }
            }
        }
        ndarray.setFloat(0, 1, 2, 2);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    if (i == 1 && j == 2 && k == 2) {
                        assertEquals(0, ndarray.getFloat(i, j, k));
                    } else {
                        assertEquals(data[i][j][k], ndarray.getFloat(i, j, k));
                    }
                }
            }
        }
    }

    @Test
    void testCreateDoubleNdArrayAndSetDouble() {
        double[][][] data = new double[][][]{
                {
                        {1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9}
                },
                {
                        {10, 11, 12},
                        {13, 14, 15},
                        {16, 17, 18}
                }
        };
        ITensor ndarray = sciCore.ndarray(data);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    assertEquals(data[i][j][k], ndarray.getDouble(i, j, k));
                }
            }
        }
        ndarray.setDouble(0, 1, 2, 2);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    if (i == 1 && j == 2 && k == 2) {
                        assertEquals(0, ndarray.getDouble(i, j, k));
                    } else {
                        assertEquals(data[i][j][k], ndarray.getDouble(i, j, k));
                    }
                }
            }
        }
    }

    @Test
    void testCreateBooleanNdArrayAndSetBoolean() {
        boolean[][][] data = new boolean[][][]{
                {
                        {true, false, true},
                        {false, true, false},
                        {true, false, true}
                },
                {
                        {false, true, false},
                        {true, false, true},
                        {false, true, false}
                }
        };
        ITensor ndarray = sciCore.ndarray(data);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    assertEquals(data[i][j][k], ndarray.getBoolean(i, j, k));
                }
            }
        }
        ndarray.setBoolean(false, 1, 2, 2);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    if (i == 1 && j == 2 && k == 2) {
                        assertFalse(ndarray.getBoolean(i, j, k));
                    } else {
                        assertEquals(data[i][j][k], ndarray.getBoolean(i, j, k));
                    }
                }
            }
        }
    }

    @Test
    void matmul_test_float_by_float_2x2by2x2() {
        ITensor matrixA = sciCore.matrix(new float[][]{{1, 2}, {3, 4}});
        ITensor matrixB = sciCore.matrix(new float[][]{{5, 6}, {7, 8}});
        ITensor result = matrixA.matmul(matrixB);
        assertEquals(sciCore.matrix(new float[][]{{19, 22}, {43, 50}}), result);
    }

    @Test
    void matmul_test_float_by_float__2x3by2x3_failure() {
        ITensor matrixA = sciCore.matrix(new float[][]{{1, 2, 3}, {4, 5, 6}});
        ITensor matrixB = sciCore.matrix(new float[][]{{7, 8, 9}, {10, 11, 12}});
        assertThrows(IllegalArgumentException.class, () -> matrixA.matmul(matrixB));
    }

    @Test
    void matmul_test_float_by_float__3d_failure() {
        ITensor matrixA = sciCore.ndarray(new float[3][4][5]);
        ITensor matrixB = sciCore.ndarray(new float[8][9][10]);
        assertThrows(IllegalArgumentException.class, () -> matrixA.matmul(matrixB));
    }

    @Test
    void matmul_test_float_by_float__2x3by3x2() {
        ITensor matrixA = sciCore.matrix(new float[][]{{1, 2, 3}, {4, 5, 6}});
        ITensor matrixB = sciCore.matrix(new float[][]{{7, 8}, {9, 10}, {11, 12}});
        ITensor result = matrixA.matmul(matrixB);
        assertEquals(sciCore.matrix(new float[][]{{58, 64}, {139, 154}}), result);
    }

    @Test
    void matmul_test_float_by_float__withDimView() {
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
    void matmul_test_float_by_float__withJvmTensor() {
        SciCore jvmSciCore = new SciCore();
        jvmSciCore.setBackend(ISciCore.BackendType.JVM);

        ITensor matrixA = sciCore.matrix(new float[][]{{1, 2}, {3, 4}});
        ITensor matrixB = jvmSciCore.matrix(new float[][]{{5, 6}, {7, 8}});
        ITensor result = matrixA.matmul(matrixB);
        assertEquals(sciCore.matrix(new float[][]{{19, 22}, {43, 50}}), result);
    }

    // TODO: OPTIMIZE JvmTensor @ CudaTensor

    @Test
    void matmul_test_double_by_double_2x2by2x2() {
        ITensor matrixA = sciCore.matrix(new double[][]{{1, 2}, {3, 4}});
        ITensor matrixB = sciCore.matrix(new double[][]{{5, 6}, {7, 8}});
        ITensor result = matrixA.matmul(matrixB);
        assertEquals(sciCore.matrix(new double[][]{{19, 22}, {43, 50}}), result);
    }

    @Test
    void matmul_test_double_by_double__2x3by2x3_failure() {
        ITensor matrixA = sciCore.matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
        ITensor matrixB = sciCore.matrix(new double[][]{{7, 8, 9}, {10, 11, 12}});
        assertThrows(IllegalArgumentException.class, () -> matrixA.matmul(matrixB));
    }

    @Test
    void matmul_test_double_by_double__3d_failure() {
        ITensor matrixA = sciCore.ndarray(new double[3][4][5]);
        ITensor matrixB = sciCore.ndarray(new double[8][9][10]);
        assertThrows(IllegalArgumentException.class, () -> matrixA.matmul(matrixB));
    }

    @Test
    void matmul_test_double_by_double__2x3by3x2() {
        ITensor matrixA = sciCore.matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
        ITensor matrixB = sciCore.matrix(new double[][]{{7, 8}, {9, 10}, {11, 12}});
        ITensor result = matrixA.matmul(matrixB);
        assertEquals(sciCore.matrix(new double[][]{{58, 64}, {139, 154}}), result);
    }

    @Test
    void matmul_test_double_by_double__withDimView() {
        ITensor bigNdArrayA = sciCore.ndarray(new double[][][]{
                {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                {{10, 11, 12}, {13, 14, 15}, {16, 17, 18}}
        });
        ITensor matrixA = bigNdArrayA.getView(0);
        ITensor matrixB = bigNdArrayA.getView(1);

        assertEquals(sciCore.matrix(new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}), matrixA);
        assertEquals(sciCore.matrix(new double[][]{{10, 11, 12}, {13, 14, 15}, {16, 17, 18}}), matrixB);

        ITensor result = matrixA.matmul(matrixB);
        assertEquals(sciCore.matrix(new double[][]{{84, 90, 96}, {201, 216, 231}, {318, 342, 366}}), result);
    }

    @Test
    void matmul_test_double_by_double__withJvmTensor() {
        SciCore jvmSciCore = new SciCore();
        jvmSciCore.setBackend(ISciCore.BackendType.JVM);

        ITensor matrixA = sciCore.matrix(new double[][]{{1, 2}, {3, 4}});
        ITensor matrixB = jvmSciCore.matrix(new double[][]{{5, 6}, {7, 8}});
        ITensor result = matrixA.matmul(matrixB);
        assertEquals(sciCore.matrix(new double[][]{{19, 22}, {43, 50}}), result);
    }

    @Test
    void matmul_test_float_by_double_2x2by2x2() {
        ITensor matrixA = sciCore.matrix(new float[][]{{1, 2}, {3, 4}});
        ITensor matrixB = sciCore.matrix(new double[][]{{5, 6}, {7, 8}});
        ITensor result = matrixA.matmul(matrixB);
        assertEquals(sciCore.matrix(new double[][]{{19, 22}, {43, 50}}), result);
    }

    @Test
    void matmul_test_float_by_double__2x3by2x3_failure() {
        ITensor matrixA = sciCore.matrix(new float[][]{{1, 2, 3}, {4, 5, 6}});
        ITensor matrixB = sciCore.matrix(new double[][]{{7, 8, 9}, {10, 11, 12}});
        assertThrows(IllegalArgumentException.class, () -> matrixA.matmul(matrixB));
    }

    @Test
    void matmul_test_float_by_double__3d_failure() {
        ITensor matrixA = sciCore.ndarray(new float[3][4][5]);
        ITensor matrixB = sciCore.ndarray(new double[8][9][10]);
        assertThrows(IllegalArgumentException.class, () -> matrixA.matmul(matrixB));
    }

    @Test
    void matmul_test_float_by_double__2x3by3x2() {
        ITensor matrixA = sciCore.matrix(new float[][]{{1, 2, 3}, {4, 5, 6}});
        ITensor matrixB = sciCore.matrix(new double[][]{{7, 8}, {9, 10}, {11, 12}});
        ITensor result = matrixA.matmul(matrixB);
        assertEquals(sciCore.matrix(new double[][]{{58, 64}, {139, 154}}), result);
    }

    @Test
    void matmul_test_float_by_double__withDimView() {
        ITensor bigNdArrayA = sciCore.ndarray(new float[][][]{
                {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                {{10, 11, 12}, {13, 14, 15}, {16, 17, 18}}
        });
        ITensor matrixA = bigNdArrayA.getView(0);
        ITensor matrixB = bigNdArrayA.getView(1);

        assertEquals(sciCore.matrix(new float[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}), matrixA);
        assertEquals(sciCore.matrix(new double[][]{{10, 11, 12}, {13, 14, 15}, {16, 17, 18}}), matrixB);

        ITensor result = matrixA.matmul(matrixB);
        assertEquals(sciCore.matrix(new double[][]{{84, 90, 96}, {201, 216, 231}, {318, 342, 366}}), result);
    }

    @Test
    void matmul_test_float_by_double__withJvmTensor() {
        SciCore jvmSciCore = new SciCore();
        jvmSciCore.setBackend(ISciCore.BackendType.JVM);

        ITensor matrixA = sciCore.matrix(new float[][]{{1, 2}, {3, 4}});
        ITensor matrixB = jvmSciCore.matrix(new double[][]{{5, 6}, {7, 8}});
        ITensor result = matrixA.matmul(matrixB);
        assertEquals(sciCore.matrix(new double[][]{{19, 22}, {43, 50}}), result);
    }

    @Test
    void matmul_test_double_by_float_2x2by2x2() {
        ITensor matrixA = sciCore.matrix(new double[][]{{1, 2}, {3, 4}});
        ITensor matrixB = sciCore.matrix(new float[][]{{5, 6}, {7, 8}});
        ITensor result = matrixA.matmul(matrixB);
        assertEquals(sciCore.matrix(new double[][]{{19, 22}, {43, 50}}), result);
    }

    @Test
    void matmul_test_double_by_float__2x3by2x3_failure() {
        ITensor matrixA = sciCore.matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
        ITensor matrixB = sciCore.matrix(new float[][]{{7, 8, 9}, {10, 11, 12}});
        assertThrows(IllegalArgumentException.class, () -> matrixA.matmul(matrixB));
    }

    @Test
    void matmul_test_double_by_float__3d_failure() {
        ITensor matrixA = sciCore.ndarray(new double[3][4][5]);
        ITensor matrixB = sciCore.ndarray(new float[8][9][10]);
        assertThrows(IllegalArgumentException.class, () -> matrixA.matmul(matrixB));
    }

    @Test
    void matmul_test_double_by_float__2x3by3x2() {
        ITensor matrixA = sciCore.matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
        ITensor matrixB = sciCore.matrix(new float[][]{{7, 8}, {9, 10}, {11, 12}});
        ITensor result = matrixA.matmul(matrixB);
        assertEquals(sciCore.matrix(new double[][]{{58, 64}, {139, 154}}), result);
    }

    @Test
    void matmul_test_double_by_float__withDimView() {
        ITensor bigNdArrayA = sciCore.ndarray(new double[][][]{
                {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                {{10, 11, 12}, {13, 14, 15}, {16, 17, 18}}
        });
        ITensor matrixA = bigNdArrayA.getView(0);
        ITensor matrixB = bigNdArrayA.getView(1);

        assertEquals(sciCore.matrix(new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}), matrixA);
        assertEquals(sciCore.matrix(new double[][]{{10, 11, 12}, {13, 14, 15}, {16, 17, 18}}), matrixB);

        ITensor result = matrixA.matmul(matrixB);
        assertEquals(sciCore.matrix(new double[][]{{84, 90, 96}, {201, 216, 231}, {318, 342, 366}}), result);
    }

    @Test
    void matmul_test_double_by_float__withJvmTensor() {
        SciCore jvmSciCore = new SciCore();
        jvmSciCore.setBackend(ISciCore.BackendType.JVM);

        ITensor matrixA = sciCore.matrix(new double[][]{{1, 2}, {3, 4}});
        ITensor matrixB = jvmSciCore.matrix(new float[][]{{5, 6}, {7, 8}});
        ITensor result = matrixA.matmul(matrixB);
        assertEquals(sciCore.matrix(new double[][]{{19, 22}, {43, 50}}), result);
    }

}
