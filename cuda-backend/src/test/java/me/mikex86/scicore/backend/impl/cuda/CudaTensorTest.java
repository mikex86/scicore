package me.mikex86.scicore.backend.impl.cuda;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.SciCore;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import static me.mikex86.scicore.ITensor.EPSILON;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertThrows;


@TestInstance(TestInstance.Lifecycle.PER_CLASS)
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


    Stream<Arguments> getMatmul_test_2x2by2x2Data() {
        double[][] a = {{1, 2}, {3, 4}};
        double[][] b = {{5, 6}, {7, 8}};
        double[][] c = {{19, 22}, {43, 50}};
        return allNumericDataTypeVariants(a, b, c);
    }

    DataType[] allDataTypes = {
            DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64,
            DataType.FLOAT32, DataType.FLOAT64
    };

    private Stream<Arguments> allNumericDataTypeVariants(Object a, Object b, Object c) {
        ITensor aMatrix = sciCore.ndarray(a);
        ITensor bMatrix = sciCore.ndarray(b);
        ITensor cMatrix = sciCore.ndarray(c);
        List<Arguments> arguments = new ArrayList<>(allDataTypes.length * allDataTypes.length);
        for (DataType dataTypeA : allDataTypes) {
            for (DataType dataTypeB : allDataTypes) {
                DataType resultDataType = DataType.getLarger(dataTypeA, dataTypeB);
                arguments.add(Arguments.of(aMatrix.cast(dataTypeA), bMatrix.cast(dataTypeB), cMatrix.cast(resultDataType)));
            }
        }
        return arguments.stream();
    }

    private Stream<Arguments> allNumericDataTypeVariants(Object a, Object b) {
        ITensor aMatrix = sciCore.ndarray(a);
        ITensor bMatrix = sciCore.ndarray(b);
        List<Arguments> arguments = new ArrayList<>(allDataTypes.length * allDataTypes.length);
        for (DataType dataTypeA : allDataTypes) {
            for (DataType dataTypeB : allDataTypes) {
                arguments.add(Arguments.of(aMatrix.cast(dataTypeA), bMatrix.cast(dataTypeB)));
            }
        }
        return arguments.stream();
    }

    @ParameterizedTest
    @MethodSource("getMatmul_test_2x2by2x2Data")
    void matmul_test_2x2by2x2(ITensor a, ITensor b, ITensor c) {
        ITensor result = a.matmul(b);
        assertEquals(c, result);
    }

    Stream<Arguments> getMatmul_test__2x3by2x3_failureData() {
        double[][] a = {{1, 2, 3}, {4, 5, 6}};
        double[][] b = {{7, 8, 9}, {10, 11, 12}};
        return allNumericDataTypeVariants(a, b);
    }

    @ParameterizedTest
    @MethodSource("getMatmul_test__2x3by2x3_failureData")
    void matmul_test__2x3by2x3_failure(ITensor a, ITensor b) {
        assertThrows(IllegalArgumentException.class, () -> a.matmul(b));
    }

    Stream<Arguments> getMatmul_test__3d_failureData() {
        double[][][] a = new double[3][4][5];
        double[][][] b = new double[8][9][10];
        return allNumericDataTypeVariants(a, b);
    }

    @ParameterizedTest
    @MethodSource("getMatmul_test__3d_failureData")
    void matmul_test__3d_failure(ITensor a, ITensor b) {
        assertThrows(IllegalArgumentException.class, () -> a.matmul(b));
    }

    Stream<Arguments> getMatmul_test__2x3by3x2Data() {
        double[][] a = {{1, 2, 3}, {4, 5, 6}};
        double[][] b = {{7, 8}, {9, 10}, {11, 12}};
        double[][] c = {{58, 64}, {139, 154}};
        return allNumericDataTypeVariants(a, b, c);
    }

    @ParameterizedTest
    @MethodSource("getMatmul_test__2x3by3x2Data")
    void matmul_test__2x3by3x2(ITensor a, ITensor b, ITensor c) {
        ITensor result = a.matmul(b);
        assertEquals(c, result);
    }

    Stream<Arguments> getMatmul_test__withDimViewData() {
        double[][][] a = {
                {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                {{10, 11, 12}, {13, 14, 15}, {16, 17, 18}}
        };
        double[][][] b = {
                {{19, 20, 21}, {22, 23, 24}, {25, 26, 27}},
                {{28, 29, 30}, {31, 32, 33}, {34, 35, 36}}
        };
        double[][] result = {{192, 198, 204}, {471, 486, 501}, {750, 774, 798}};
        return allNumericDataTypeVariants(a, b, result).map(args -> {
            ITensor aMatrix = (ITensor) args.get()[0];
            ITensor bMatrix = (ITensor) args.get()[1];
            ITensor cMatrix = (ITensor) args.get()[2];
            return Arguments.of(aMatrix.getView(0), bMatrix.getView(1), cMatrix);
        });
    }

    @ParameterizedTest
    @MethodSource("getMatmul_test__withDimViewData")
    void matmul_test__withDimView(ITensor a, ITensor b, ITensor c) {
        ITensor result = a.matmul(b);
        assertEquals(c, result);
    }

    Stream<Arguments> getMatmul_test__withJvmTensorData() {
        SciCore jvmSciCore = new SciCore();
        jvmSciCore.setBackend(ISciCore.BackendType.JVM);
        double[][] a = {{1, 2}, {3, 4}};
        double[][] b = {{5, 6}, {7, 8}};
        double[][] c = {{19, 22}, {43, 50}};
        Stream<Arguments> argumentsStream = allNumericDataTypeVariants(a, b, c);
        return argumentsStream.map(arguments -> {
            Object[] objects = arguments.get();
            ITensor aMatrix = (ITensor) objects[0];
            ITensor bMatrix = (ITensor) objects[1];
            ITensor cMatrix = (ITensor) objects[2];
            ITensor bMatrixJvm = jvmSciCore.zeros(bMatrix.getDataType(), bMatrix.getShape());
            bMatrixJvm.setContents(bMatrix);
            return Arguments.of(aMatrix, bMatrixJvm, cMatrix);
        });
    }

    @ParameterizedTest
    @MethodSource("getMatmul_test__withJvmTensorData")
    void matmul_test__withJvmTensor(ITensor a, ITensor b, ITensor c) {
        ITensor result = a.matmul(b);
        assertEquals(c, result);
    }

    // TODO: TEST BOOLEAN MULTIPLY

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
