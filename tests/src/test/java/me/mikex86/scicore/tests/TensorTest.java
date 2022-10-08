package me.mikex86.scicore.tests;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.SciCore;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;


class TensorTest {

    private static final float EPSILON = 1E-6f;

    ISciCore sciCore;

    TensorTest(@NotNull ISciCore.BackendType backendType) {
        this.sciCore = new SciCore();
        this.sciCore.setBackend(backendType);
        this.sciCore.disableBackendFallback();
    }

    @Nested
    class CreateArrays {
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
    }


    @Nested
    class SetValues {

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
    }


    @Nested
    class CreateMatrices {
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
    }


    @Nested
    class CreateMatricesAndSetValues {

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
            Assertions.assertFalse(matrix.getBoolean(1, 1));
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    if (i == 1 && j == 1) {
                        continue;
                    }
                    assertEquals(data[i][j], matrix.getBoolean(i, j));
                }
            }
        }
    }


    @Nested
    class CreateNdArrays {

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
    }

    @Nested
    class CreateNdArrayAndSetValues {

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
                            Assertions.assertFalse(ndarray.getBoolean(i, j, k));
                        } else {
                            assertEquals(data[i][j][k], ndarray.getBoolean(i, j, k));
                        }
                    }
                }
            }
        }
    }

    @Test
    void getView() {
        ITensor matrix = sciCore.matrix(new float[][]{{12.4f, 16.3f}, {1.2f, 9.1f}, {7.3f, 3.4f}});

        Assertions.assertDoesNotThrow(() -> matrix.getView(0));
        Assertions.assertDoesNotThrow(() -> matrix.getView(1));
        Assertions.assertDoesNotThrow(() -> matrix.getView(2));
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
        Assertions.assertArrayEquals(shape, ndArray.getShape());
    }

    @NotNull
    static Stream<Object> testCreateInvalidNdArrayShapeData() {
        return Stream.of(
                new byte[][]{{1, 2}, {3}},
                new byte[][]{{1, 2}, {3, 4, 5}},
                new byte[][]{{1, 2}, {3, 4}, {5, 6, 7}},
                new byte[][]{{1, 2}, {3, 4}, {5, 6}, {7, 8, 9}},
                new short[][]{{1, 2}, {3}},
                new short[][]{{1, 2}, {3, 4, 5}},
                new short[][]{{1, 2}, {3, 4}, {5, 6, 7}},
                new short[][]{{1, 2}, {3, 4}, {5, 6}, {7, 8, 9}},
                new float[][]{{1, 2}, {3}},
                new float[][]{{1, 2}, {3, 4, 5}},
                new float[][]{{1, 2}, {3, 4}, {5, 6, 7}},
                new float[][]{{1, 2}, {3, 4}, {5, 6}, {7, 8, 9}},
                new double[][]{{1, 2}, {3}},
                new double[][]{{1, 2}, {3, 4, 5}},
                new double[][]{{1, 2}, {3, 4}, {5, 6, 7}},
                new double[][]{{1, 2}, {3, 4}, {5, 6}, {7, 8, 9}}
        );
    }

    @ParameterizedTest
    @MethodSource("testCreateInvalidNdArrayShapeData")
    void testCreateInvalidNdArray_failure(Object javaArray) {
        assertThrows(IllegalArgumentException.class, () -> sciCore.ndarray(javaArray));
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


    @Nested
    class ToString {
        @Test
        void toString_test_scalar() {
            ITensor scalar = sciCore.scalar(42.0f);
            String className = scalar.getClass().getSimpleName();
            assertEquals(className + "(dtype=FLOAT32, shape=[], isScalar=true, data=42.0)", scalar.toString());
        }

        @Test
        void toString_test_1dArray() {
            ITensor array = sciCore.array(new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
            String className = array.getClass().getSimpleName();
            assertEquals(className + "(dtype=FLOAT32, shape=[5], data=[1.0, 2.0, 3.0, 4.0, 5.0])", array.toString());
        }

        @Test
        void toString_test_2dMatrix() {
            ITensor matrix = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
            String className = matrix.getClass().getSimpleName();
            assertEquals(className + "(dtype=FLOAT32, shape=[2, 3], data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])", matrix.toString());
        }

        @Test
        void toString_test_2dMatrix_large() {
            ITensor matrix = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f},
                    {16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f}});
            String className = matrix.getClass().getSimpleName();
            assertEquals(className + """
                    (dtype=FLOAT32, shape=[2, 15], data=
                    \t[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                    \t [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0]])""", matrix.toString());
        }

        @Test
        void toString_test_view() {
            ITensor matrix = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
            ITensor view = matrix.getReshapedView(new long[]{3, 2}, new long[]{1, 3});
            String className = view.getClass().getSimpleName();
            assertEquals(className + "(dtype=FLOAT32, shape=[3, 2], data=[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])", view.toString());
        }
    }

    @Test
    void exp() {
        ITensor matrix = sciCore.matrix(new float[][]{{3.8f, 46.3f}, {2.7f, 1.9f}, {3.7f, 1.7f}});
        ITensor exp = matrix.exp();
        assertEquals((float) Math.exp(3.8f), exp.getFloat(0, 0), EPSILON);
        assertEquals((float) Math.exp(46.3f), exp.getFloat(0, 1), EPSILON);
    }


    @Nested
    @TestInstance(TestInstance.Lifecycle.PER_CLASS)
    class Matmul {

        private ISciCore jvmSciCore;

        @BeforeAll
        void setUp() {
            jvmSciCore = new SciCore();
            jvmSciCore.setBackend(ISciCore.BackendType.JVM);
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
            ITensor aMatrix = jvmSciCore.ndarray(a);
            ITensor bMatrix = jvmSciCore.ndarray(b);
            ITensor cMatrix = jvmSciCore.ndarray(c);
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
            ITensor aMatrix = jvmSciCore.ndarray(a);
            ITensor bMatrix = jvmSciCore.ndarray(b);
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
        void matmul_test_2x2by2x2(final ITensor a, final ITensor b, ITensor c) {
            ITensor result = a.matmul(b);
            assertEquals(c, result);
        }

        Stream<Arguments> getMatmul_test_2x2_by_2x2_transposeAData() {
            double[][] a = {{1, 2}, {3, 4}};
            double[][] b = {{5, 6}, {7, 8}};
            double[][] c = {{26, 30}, {38, 44}};
            return allNumericDataTypeVariants(a, b, c);
        }

        @ParameterizedTest
        @MethodSource("getMatmul_test_2x2_by_2x2_transposeAData")
        void matmul_test_2x2_by_2x2_transposeA(final ITensor a, final ITensor b, ITensor c) {
            ITensor result = a.matmul(b, true, false);
            assertEquals(c, result);
        }

        Stream<Arguments> getMatmul_test_2x2_by_2x2_transposeBData() {
            double[][] a = {{1, 2}, {3, 4}};
            double[][] b = {{5, 6}, {7, 8}};
            double[][] c = {{17, 23}, {39, 53}};
            return allNumericDataTypeVariants(a, b, c);
        }

        @ParameterizedTest
        @MethodSource("getMatmul_test_2x2_by_2x2_transposeBData")
        void matmul_test_2x2_by_2x2_transposeB(final ITensor a, final ITensor b, ITensor c) {
            ITensor result = a.matmul(b, false, true);
            assertEquals(c, result);
        }

        Stream<Arguments> getMatmul_test_2x2_by_2x2_transposeA_and_transposeBData() {
            double[][] a = {{1, 2}, {3, 4}};
            double[][] b = {{5, 6}, {7, 8}};
            double[][] c = {{23, 31}, {34, 46}};
            return allNumericDataTypeVariants(a, b, c);
        }

        @ParameterizedTest
        @MethodSource("getMatmul_test_2x2_by_2x2_transposeA_and_transposeBData")
        void matmul_test_2x2_by_2x2_transposeA_and_transposeB(final ITensor a, final ITensor b, ITensor c) {
            ITensor result = a.matmul(b, true, true);
            assertEquals(c, result);
        }

        Stream<Arguments> getMatmul_test_2x3by2x3__failureData() {
            double[][] a = {{1, 2, 3}, {4, 5, 6}};
            double[][] b = {{7, 8, 9}, {10, 11, 12}};
            return allNumericDataTypeVariants(a, b);
        }

        @ParameterizedTest
        @MethodSource("getMatmul_test_2x3by2x3__failureData")
        void matmul_test_2x3by2x3__failure(final ITensor a, final ITensor b) {
            assertThrows(IllegalArgumentException.class, () -> a.matmul(b));
        }

        Stream<Arguments> getMatmul_test_3d_failureData() {
            double[][][] a = new double[3][4][5];
            double[][][] b = new double[8][9][10];
            return allNumericDataTypeVariants(a, b);
        }

        @ParameterizedTest
        @MethodSource("getMatmul_test_3d_failureData")
        void matmul_test_3d__failure(final ITensor a, final ITensor b) {
            assertThrows(IllegalArgumentException.class, () -> a.matmul(b));
        }

        Stream<Arguments> getMatmul_test_2x3by3x2Data() {
            double[][] a = {{1, 2, 3}, {4, 5, 6}};
            double[][] b = {{7, 8}, {9, 10}, {11, 12}};
            double[][] c = {{58, 64}, {139, 154}};
            return allNumericDataTypeVariants(a, b, c);
        }

        @ParameterizedTest
        @MethodSource("getMatmul_test_2x3by3x2Data")
        void matmul_test_2x3by3x2(final ITensor a, final ITensor b, ITensor c) {
            ITensor result = a.matmul(b);
            assertEquals(c, result);
        }

        Stream<Arguments> getMatmul_test_2x3by3x2_transposeAData() {
            double[][] a = {{1, 2, 3}, {4, 5, 6}};
            double[][] b = {{7, 8}, {9, 10}, {11, 12}};
            return allNumericDataTypeVariants(a, b);
        }

        @ParameterizedTest
        @MethodSource("getMatmul_test_2x3by3x2_transposeAData")
        void matmul_test_2x3by3x2_transposeA_failure(final ITensor a, final ITensor b) {
            assertThrows(IllegalArgumentException.class, () -> a.matmul(b, true, false));
        }

        Stream<Arguments> getMatmul_test_2x3by3x2_transposeBData() {
            double[][] a = {{1, 2, 3}, {4, 5, 6}};
            double[][] b = {{7, 8}, {9, 10}, {11, 12}};
            return allNumericDataTypeVariants(a, b);
        }

        @ParameterizedTest
        @MethodSource("getMatmul_test_2x3by3x2_transposeBData")
        void matmul_test_2x3by3x2_transposeB__failure(final ITensor a, final ITensor b) {
            assertThrows(IllegalArgumentException.class, () -> a.matmul(b, false, true));
        }

        Stream<Arguments> getMatmul_test_2x3by3x2_transposeA_and_transposeBData() {
            double[][] a = {{1, 2, 3}, {4, 5, 6}};
            double[][] b = {{7, 8}, {9, 10}, {11, 12}};
            double[][] c = {{39, 49, 59}, {54, 68, 82}, {69, 87, 105}};
            return allNumericDataTypeVariants(a, b, c);
        }

        @ParameterizedTest
        @MethodSource("getMatmul_test_2x3by3x2_transposeA_and_transposeBData")
        void matmul_test_2x3by3x2_transposeA_and_transposeB(final ITensor a, final ITensor b, ITensor c) {
            ITensor result = a.matmul(b, true, true);
            assertEquals(c, result);
        }

        Stream<Arguments> getMatmul_test_withDimViewData() {
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
        @MethodSource("getMatmul_test_withDimViewData")
        void matmul_test_withDimView(final ITensor a, final ITensor b, ITensor c) {
            ITensor result = a.matmul(b);
            assertEquals(c, result);
        }

        Stream<Arguments> getMatmul_test_withJvmTensorData() {
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
        @MethodSource("getMatmul_test_withJvmTensorData")
        void matmul_test_withJvmTensor(final ITensor a, final ITensor b, ITensor c) {
            ITensor result = a.matmul(b);
            assertEquals(c, result);
        }
    }


    @Nested
    class Divide {

        @Test
        void divide_test_2x2x2by2x2() {
            ITensor a = sciCore.ndarray(new float[][][]{{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}});
            ITensor b = sciCore.matrix(new float[][]{{5.0f, 6.0f}, {7.0f, 8.0f}});
            ITensor result = a.divide(b);
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
        void divide_test_2x2by2x1() {
            ITensor a = sciCore.matrix(new double[][]{{4.4701180e+01, 1.2818411e+20}, {1.4879732e+01, 6.6858945e+00}});
            ITensor b = sciCore.matrix(new double[][]{{1.2818411e+20}, {2.1565626e+01}});
            ITensor result = a.divide(b);
            assertEquals(3.4872637e-19, result.getDouble(0, 0), EPSILON);
            assertEquals(1.0000000e+00, result.getDouble(0, 1), EPSILON);
            assertEquals(6.8997449e-01, result.getDouble(1, 0), EPSILON);
            assertEquals(3.1002551e-01, result.getDouble(1, 1), EPSILON);
        }
    }


    class Softmax {
        @Test
        void softmax_test_dim1() {
            ITensor matrix = sciCore.matrix(new float[][]{{3.8f, 46.3f}, {2.7f, 1.9f}});
            ITensor softmax = matrix.softmax(1);
            assertEquals(sciCore.matrix(new float[][]{{3.4872616e-19f, 1.0000000e+00f}, {6.8997449e-01f, 3.1002548e-01f}}), softmax);
        }
    }


    @Nested
    class ReduceSum {

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
            Assertions.assertArrayEquals(new long[]{3, 4}, sum.getShape());
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
            Assertions.assertArrayEquals(new long[]{5, 3}, sum.getShape());
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
            Assertions.assertArrayEquals(new long[]{5, 3, 1}, sum.getShape());
            assertEquals(sciCore.ndarray(new float[][][]{
                    {{10 + 12 + 14 + 16}, {12 + 14 + 16 + 18}, {14 + 16 + 18 + 20}},
                    {{16 + 18 + 20 + 22}, {18 + 20 + 22 + 24}, {20 + 22 + 24 + 26}},
                    {{22 + 24 + 26 + 28}, {24 + 26 + 28 + 30}, {26 + 28 + 30 + 32}},
                    {{28 + 30 + 32 + 34}, {30 + 32 + 34 + 36}, {32 + 34 + 36 + 38}},
                    {{34 + 36 + 38 + 40}, {36 + 38 + 40 + 42}, {38 + 40 + 42 + 44}}}), sum);
        }
    }


    @Nested
    class Plus {

        @Test
        void plus_test_1x1by1x1_success() {
            ITensor a = sciCore.matrix(new float[][]{{1.0f}});
            ITensor b = sciCore.matrix(new float[][]{{2.0f}});
            ITensor result = a.plus(b);
            assertEquals(3.0f, result.getFloat(0, 0), EPSILON);
        }

        @Test
        void plus_test_2x2by2x1_broadcast_success() {
            ITensor a = sciCore.matrix(new float[][]{{1, 2}, {3, 4}});
            ITensor b = sciCore.matrix(new float[][]{{5}, {6}});
            ITensor result = a.plus(b);
            assertEquals(sciCore.matrix(new float[][]{{6, 7}, {9, 10}}), result);
        }

        @Test
        void plus_test_2x2by2x1x1_broadcast_success() {
            ITensor a = sciCore.matrix(new float[][]{{1, 2}, {3, 4}});
            ITensor b = sciCore.ndarray(new float[][][]{{{5}}, {{6}}});
            ITensor result = a.plus(b);
            assertEquals(sciCore.ndarray(new float[][][]{{{6, 7}, {8, 9}}, {{7, 8}, {9, 10}}}), result);
        }

        @Test
        void plus_test_2by2x1x1_broadcast_success() {
            ITensor a = sciCore.array(new float[]{1, 2});
            ITensor b = sciCore.ndarray(new float[][][]{{{5}}, {{6}}});
            ITensor result = a.plus(b);
            assertEquals(sciCore.ndarray(new float[][][]{{{6, 7}}, {{7, 8}}}), result);
        }

        @Test
        void plus_test_2x2by2x1x2_broadcast_success() {
            ITensor a = sciCore.matrix(new float[][]{{1, 2}, {3, 4}});
            ITensor b = sciCore.ndarray(new float[][][]{{{6, 8}}, {{11, 13}}});
            ITensor result = a.plus(b);
            assertEquals(sciCore.ndarray(new float[][][]{
                    {{7, 10}, {9, 12}},
                    {{12, 15}, {14, 17}}}), result);
        }

        @Test
        void plus_test_2x3by2x1x2x1x3_broadcast_success() {
            ITensor a = sciCore.matrix(new float[][]{{1, 2, 3}, {4, 5, 6}});
            ITensor b = sciCore.ndarray(new float[][][][][]{{{{{7, 11, 15}}, {{16, 19, 20}}}}, {{{{21, 22, 24}}, {{27, 28, 30}}}}});
            ITensor result = a.plus(b);
            assertEquals(sciCore.ndarray(new float[][][][][]{{{{{8, 13, 18}, {11, 16, 21}}, {{17, 21, 23}, {20, 24, 26}}}}, {{{{22, 24, 27}, {25, 27, 30}}, {{28, 30, 33}, {31, 33, 36}}}}}), result);
        }

        @Test
        void plus_tensor_2x3by1x2x1_broadcast_success() {
            ITensor a = sciCore.matrix(new float[][]{{1, 2, 3}, {4, 5, 6}});
            ITensor b = sciCore.ndarray(new float[][][]{{{5}, {6}}});
            ITensor result = a.plus(b);
            assertEquals(sciCore.ndarray(new float[][][]{
                    {{6, 7, 8}, {10, 11, 12}}
            }), result);
        }

        @Test
        void plus_test_3x2by1x1_success() {
            ITensor a = sciCore.matrix(new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
            ITensor b = sciCore.matrix(new float[][]{{2.0f}});
            ITensor result = a.plus(b);
            assertEquals(sciCore.matrix(new float[][]{{3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}}), result);
        }

        @Test
        void plus_test_5x3x2by3x2_2d_broadcast_success() {
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
        void plus_test_2x3x4by3x3x4_broadcastImpossible_failure() {
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
        void plus_test_3dby3d_broadcast_firstDimIsOne_success() {
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
    }

    @Nested
    class MinusTest {
        @Test
        void minus_test_1x1by1x1_success() {
            ITensor a = sciCore.matrix(new float[][]{{1.0f}});
            ITensor b = sciCore.matrix(new float[][]{{2.0f}});
            ITensor result = a.minus(b);
            assertEquals(-1.0f, result.getFloat(0, 0), EPSILON);
        }

        @Test
        void minus_test_2x2by2x2_success() {
            ITensor a = sciCore.matrix(new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}});
            ITensor b = sciCore.matrix(new float[][]{{2.0f, 3.0f}, {4.0f, 5.0f}});
            ITensor result = a.minus(b);
            assertEquals(sciCore.matrix(new float[][]{{-1.0f, -1.0f}, {-1.0f, -1.0f}}), result);
        }

        @Test
        void minus_test_2x2by2x1_success() {
            ITensor a = sciCore.matrix(new float[][]{{1, 2}, {3, 4}});
            ITensor b = sciCore.matrix(new float[][]{{5}, {6}});
            ITensor result = a.minus(b);
            assertEquals(sciCore.matrix(new float[][]{{-4, -3}, {-3, -2}}), result);
        }

        @Test
        void minus_test_2x2by2_success() {
            ITensor a = sciCore.matrix(new float[][]{{1, 2}, {3, 4}});
            ITensor b = sciCore.ndarray(new float[]{5, 6});
            ITensor result = a.minus(b);
            assertEquals(sciCore.matrix(new float[][]{{-4, -4}, {-2, -2}}), result);
        }

        @Test
        void minus_test_5x3x2by3x2_2d_broadcast_success() {
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
        void minus_test_2x3x4by3x3x4_broadcastImpossible_failure() {
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
        void minus_test_3d_minus_3d_broadcast_firstDimIsOne_success() {
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
    }


    @Nested
    class MultiplyTest {

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
            ITensor b = sciCore.scalar(10f);

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

        @Test
        void multiply_test_1x1by1x1_success() {
            ITensor a = sciCore.matrix(new float[][]{{1.0f}});
            ITensor b = sciCore.matrix(new float[][]{{2.0f}});
            ITensor c = a.multiply(b);
            assertEquals(sciCore.matrix(new float[][]{{2.0f}}), c);
        }

        @Test
        void multiply_test_2x2by2x1_broadcast_success() {
            ITensor a = sciCore.matrix(new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}});
            ITensor b = sciCore.matrix(new float[][]{{2.0f}, {3.0f}});
            ITensor result = a.multiply(b);
            assertEquals(sciCore.matrix(new float[][]{{2, 4}, {9, 12}}), result);
        }

        @Test
        void multiply_test_2x2by2x1x1_broadcast_success() {
            ITensor a = sciCore.matrix(new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}});
            ITensor b = sciCore.ndarray(new float[][][]{{{2.0f}}, {{3.0f}}});
            ITensor result = a.multiply(b);
            assertEquals(sciCore.ndarray(new float[][][]{{{2, 4}, {6, 8}}, {{3, 6}, {9, 12}}}), result);
        }

        @Test
        void multiply_test_1x2x2by2x1x1_broadcast_success() {
            ITensor a = sciCore.ndarray(new float[][][]{{{1.0f, 2.0f}}, {{3.0f, 4.0f}}});
            ITensor b = sciCore.ndarray(new float[][][]{{{2.0f}}, {{3.0f}}});
            ITensor result = a.multiply(b);
            assertEquals(sciCore.ndarray(new float[][][]{{{2, 4}}, {{9, 12}}}), result);
        }

        @Test
        void plus_test_2by2x1x1_broadcast_success() {
            ITensor a = sciCore.array(new float[]{1.0f, 2.0f});
            ITensor b = sciCore.ndarray(new float[][][]{{{2.0f}}, {{3.0f}}});
            ITensor result = a.multiply(b);
            assertEquals(sciCore.ndarray(new float[][][]{{{2, 4}}, {{3, 6}}}), result);
        }

        @Test
        void multiply_test_2x2by2x1x2_broadcast_success() {
            ITensor a = sciCore.matrix(new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}});
            ITensor b = sciCore.ndarray(new float[][][]{{{2.0f, 3.0f}}, {{3.0f, 4.0f}}});
            ITensor result = a.multiply(b);
            assertEquals(sciCore.ndarray(new float[][][]{{{2.0f, 6.0f}, {6.0f, 12.0f}}, {{3.0f, 8.0f}, {9.0f, 16.0f}}}), result);
        }

        @Test
        void multiply_test_2x3by2x1x2x1x3_broadcast_success() {
            ITensor a = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
            ITensor b = sciCore.ndarray(new float[][][][][]{{{{{7, 11, 15}}, {{16, 19, 20}}}}, {{{{21, 22, 24}}, {{27, 28, 30}}}}});
            ITensor result = a.multiply(b);
            assertEquals(sciCore.ndarray(new float[][][][][]{{{{{7.0f, 22.0f, 45.0f}, {28.0f, 55.0f, 90.0f}}, {{16.0f, 38.0f, 60.0f}, {64.0f, 95.0f, 120.0f}}}}, {{{{21.0f, 44.0f, 72.0f}, {84.0f, 110.0f, 144.0f}}, {{27.0f, 56.0f, 90.0f}, {108.0f, 140.0f, 180.0f}}}}}), result);
        }

        @Test
        void multiply_tensor_2x3by1x2x1_broadcast_success() {
            ITensor a = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
            ITensor b = sciCore.ndarray(new float[][][]{{{5}, {6}}});
            ITensor result = a.multiply(b);
            assertEquals(sciCore.ndarray(new float[][][]{{{5.0f, 10.0f, 15.0f}, {24.0f, 30.0f, 36.0f}}}), result);
        }

        @Test
        void multiply_test_3x2by1x1_success() {
            ITensor a = sciCore.matrix(new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
            ITensor b = sciCore.matrix(new float[][]{{7.0f}});
            ITensor result = a.multiply(b);
            assertEquals(sciCore.matrix(new float[][]{{7.0f, 14.0f}, {21.0f, 28.0f}, {35.0f, 42.0f}}), result);
        }

        @Test
        void multiply_test_5x3x2by3x2_2d_broadcast_success() {
            ITensor a = sciCore.ndarray(new float[][][]{
                    {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
                    {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}},
                    {{13.0f, 14.0f}, {15.0f, 16.0f}, {17.0f, 18.0f}},
                    {{19.0f, 20.0f}, {21.0f, 22.0f}, {23.0f, 24.0f}},
                    {{25.0f, 26.0f}, {27.0f, 28.0f}, {29.0f, 30.0f}}
            });
            ITensor b = sciCore.ndarray(new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
            ITensor c = a.multiply(b);
            assertEquals(sciCore.ndarray(new float[][][]{
                    {{1.0f, 4.0f}, {9.0f, 16.0f}, {25.0f, 36.0f}},
                    {{7.0f, 16.0f}, {27.0f, 40.0f}, {55.0f, 72.0f}},
                    {{13.0f, 28.0f}, {45.0f, 64.0f}, {85.0f, 108.0f}},
                    {{19.0f, 40.0f}, {63.0f, 88.0f}, {115.0f, 144.0f}},
                    {{25.0f, 52.0f}, {81.0f, 112.0f}, {145.0f, 180.0f}}
            }), c);
        }

        @Test
        void multiply_test_2x3x4by3x3x4_broadcastImpossible_failure() {
            ITensor a = sciCore.ndarray(new float[][][]{
                    {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}},
                    {{13.0f, 14.0f, 15.0f, 16.0f}, {17.0f, 18.0f, 19.0f, 20.0f}, {21.0f, 22.0f, 23.0f, 24.0f}}});
            ITensor b = sciCore.ndarray(new float[][][]{
                    {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}},
                    {{13.0f, 14.0f, 15.0f, 16.0f}, {17.0f, 18.0f, 19.0f, 20.0f}, {21.0f, 22.0f, 23.0f, 24.0f}},
                    {{25.0f, 26.0f, 27.0f, 28.0f}, {29.0f, 30.0f, 31.0f, 32.0f}, {33.0f, 34.0f, 35.0f, 36.0f}}});
            assertThrows(IllegalArgumentException.class, () -> a.multiply(b));
        }

        @Test
        void multiply_test_3dby3d_broadcast_firstDimIsOne_success() {
            ITensor a = sciCore.ndarray(new float[][][]{
                    {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
                    {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}},
            });
            ITensor b = sciCore.ndarray(new float[][][]{
                    {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
            });
            ITensor c = a.multiply(b);
            assertEquals(sciCore.ndarray(new float[][][]{
                    {{1.0f, 4.0f}, {9.0f, 16.0f}, {25.0f, 36.0f}},
                    {{7.0f, 16.0f}, {27.0f, 40.0f}, {55.0f, 72.0f}}
            }), c);
        }

    }


    @Nested
    class Argmax {

        @Test
        void test_argmax_2d_dimMinus1() {
            ITensor a = sciCore.matrix(new float[][]{
                    {2, 1, 0}, {1, 0, 3}
            });
            ITensor max2 = a.argmax(-1);
            assertEquals(sciCore.scalar(5L), max2);
        }

        @Test
        void test_argmax_2d_dim0() {
            ITensor a = sciCore.matrix(new float[][]{
                    {2, 1, 0}, {1, 0, 3}
            });
            ITensor max1 = a.argmax(0);
            assertEquals(sciCore.array(new long[]{0, 0, 1}), max1);
        }

        @Test
        void test_argmax_2d_dim1() {
            ITensor a = sciCore.matrix(new float[][]{
                    {2, 1, 0}, {1, 0, 3}
            });
            ITensor max2 = a.argmax(1);
            assertEquals(sciCore.array(new long[]{0, 2}), max2);
        }
    }

    @Nested
    class Pow {

        @Test
        void test_pow_2d() {
            ITensor a = sciCore.matrix(new float[][]{
                    {2, 1, 0}, {1, 0, 3}
            });
            ITensor pow = a.pow(2f);
            assertEquals(sciCore.matrix(new float[][]{
                    {4, 1, 0}, {1, 0, 9}
            }), pow);
        }

        @Test
        void test_pow_3d() {
            ITensor a = sciCore.ndarray(new float[][][]{
                    {{2, 1, 0}, {1, 0, 3}},
                    {{2, 1, 0}, {1, 0, 3}}
            });
            ITensor pow = a.pow(2f);
            assertEquals(sciCore.ndarray(new float[][][]{
                    {{4, 1, 0}, {1, 0, 9}},
                    {{4, 1, 0}, {1, 0, 9}}
            }), pow);
        }

        @Test
        void test_pow_4d() {
            ITensor a = sciCore.ndarray(new float[][][][]{
                    {{{2, 1, 0}, {1, 0, 3}}},
                    {{{2, 1, 0}, {1, 0, 3}}}
            });
            ITensor pow = a.pow(2f);
            assertEquals(sciCore.ndarray(new float[][][][]{
                    {{{4, 1, 0}, {1, 0, 9}}},
                    {{{4, 1, 0}, {1, 0, 9}}}
            }), pow);
        }
    }

    @Nested
    class Transpose {

        @Test
        void transpose_2d() {
            ITensor a = sciCore.matrix(new float[][]{
                    {2, 1, 0}, {1, 7, 3}
            });
            ITensor transpose = a.transpose();
            assertEquals(sciCore.matrix(new float[][]{
                    {2, 1}, {1, 7}, {0, 3}
            }), transpose);
        }

        @Test
        void transpose_3d_failure() {
            ITensor a = sciCore.ndarray(new float[][][]{
                    {{2, 1, 0}, {1, 0, 3}},
                    {{2, 1, 0}, {1, 0, 3}}
            });
            assertThrows(IllegalArgumentException.class, a::transpose);
        }

    }
}