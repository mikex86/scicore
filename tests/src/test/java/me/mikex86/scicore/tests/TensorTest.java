package me.mikex86.scicore.tests;

import me.mikex86.scicore.graph.IGraphRecorder;
import me.mikex86.scicore.graph.OperationType;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.SciCore;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

@Disabled // this test is disabled because it is abstract
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
abstract class TensorTest {

    private static final float EPSILON = 1E-3f;

    ISciCore sciCore;

    TensorTest(@NotNull ISciCore.BackendType backendType) {
        this.sciCore = new SciCore();
        this.sciCore.addBackend(backendType);
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

    @Test
    void exp() {
        ITensor matrix = sciCore.matrix(new float[][]{{3.8f, 4.3f}, {2.7f, 1.9f}, {3.7f, 1.7f}});
        ITensor exp = matrix.exp();
        assertEquals(sciCore.matrix(new float[][]{{(float) Math.exp(3.8), (float) Math.exp(4.3)}, {(float) Math.exp(2.7), (float) Math.exp(1.9)}, {(float) Math.exp(3.7), (float) Math.exp(1.7)}}), exp);
    }

    @Nested
    @TestInstance(TestInstance.Lifecycle.PER_CLASS)
    class UnaryOpOpsRespectStrides {

        Stream<OperationType> allArithmeticUnaryOps() {
            return Stream.of(OperationType.values())
                    .filter(op -> op.getArity() == OperationType.Arity.UNARY &&
                                  op.getCategory() == OperationType.Category.ARITHMETIC);
        }

        @ParameterizedTest
        @MethodSource("allArithmeticUnaryOps")
        void arithmeticUnaryOps_test_respectsStrides(OperationType operationType) {
            ITensor tensor = sciCore.ndarray(new float[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
            ITensor viewed = tensor.view(new long[]{3, 3}, new long[]{1, 3});
            ITensor equalToView = sciCore.ndarray(new float[][]{{1, 4, 7}, {2, 5, 8}, {3, 6, 9}});
            assertEquals(equalToView, viewed);
            IGraphRecorder operationRecorder = sciCore.getBackend().getOperationRecorder();
            ITensor resultOfView = operationRecorder.recordOperation(operationType, sciCore.getBackend(), viewed);
            ITensor resultOfEqualToView = operationRecorder.recordOperation(operationType, sciCore.getBackend(), equalToView);
            assertEquals(resultOfEqualToView, resultOfView);
        }

        Stream<Arguments> matrixMultiplicationRespectsStridesData() {
            ITensor a = sciCore.matrix(new float[][]{{54, 33, 28}, {96, 23, 18}, {63, 9, 72}}); // (3, 3)
            ITensor b = sciCore.matrix(new float[][]{{42, 1, 9}, {59, 31, 29}, {12, 3, 7}}); // (3, 3)

            ITensor aTransposedManually = sciCore.matrix(new float[][]{{54, 96, 63}, {33, 23, 9}, {28, 18, 72}}); // (3, 3)
            ITensor bTransposedManually = sciCore.matrix(new float[][]{{42, 59, 12}, {1, 31, 3}, {9, 29, 7}}); // (3, 3)

            List<Arguments> arguments = new ArrayList<>();
            for (DataType dataTypeA : DataType.values()) {
                if (!dataTypeA.isNumeric()) {
                    continue;
                }
                for (DataType dataTypeB : DataType.values()) {
                    if (!dataTypeB.isNumeric()) {
                        continue;
                    }

                    ITensor aTransposedManuallyCast = aTransposedManually.cast(dataTypeA);
                    ITensor bTransposedManuallyCast = bTransposedManually.cast(dataTypeB);
                    for (boolean transposeAViaViews : new boolean[]{false, true}) {
                        for (boolean transposeBViaViews : new boolean[]{false, true}) {
                            for (boolean transposeAinMatmul : new boolean[]{false, true}) {
                                for (boolean transposeBinMatmul : new boolean[]{false, true}) {
                                    ITensor aCast = a.cast(dataTypeA);
                                    ITensor bCast = b.cast(dataTypeB);
                                    ITensor aTransposedViaViews = transposeAViaViews ? aCast.transpose() : aCast;
                                    assertEquals(transposeAViaViews ? aTransposedManuallyCast : aCast, aTransposedViaViews);

                                    ITensor bTransposedViaViews = transposeBViaViews ? bCast.transpose() : bCast;
                                    assertEquals(transposeBViaViews ? bTransposedManuallyCast : bCast, bTransposedViaViews);

                                    boolean netTransposeForExpectedA = transposeAViaViews ^ transposeAinMatmul;
                                    boolean netTransposeForExpectedB = transposeBViaViews ^ transposeBinMatmul;

                                    ITensor netATransposeWithManualTranspose = netTransposeForExpectedA ? aTransposedManuallyCast : aCast;
                                    ITensor netBTransposeWithManualTranspose = netTransposeForExpectedB ? bTransposedManuallyCast : bCast;

                                    ITensor expected = sciCore.matmul(netATransposeWithManualTranspose, netBTransposeWithManualTranspose);
                                    ITensor expected2 = sciCore.matmul(aCast, bCast, netTransposeForExpectedA, netTransposeForExpectedB);

                                    assertEquals(expected, expected2);

                                    arguments.add(Arguments.of(aTransposedViaViews, bTransposedViaViews, transposeAinMatmul, transposeBinMatmul, expected));
                                }
                            }
                        }
                    }
                }
            }
            return arguments.stream();
        }

        @ParameterizedTest
        @MethodSource("matrixMultiplicationRespectsStridesData")
        void matrixMultiplicationRespectsStrides(ITensor a, ITensor b, boolean transposeAInMatmul, boolean transposeBInMatmul, ITensor expected) {
            ITensor result = sciCore.matmul(a, b, transposeAInMatmul, transposeBInMatmul);
            assertEquals(expected, result);
        }

    }


    @Nested
    class Matmul {

        @Nested
        @TestInstance(TestInstance.Lifecycle.PER_CLASS)
        class Matmul2d {

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


            Stream<Arguments> getMatmul_test_2x3by3x4Data() {
                double[][] a = {{1, 2, 3}, {4, 5, 6}};
                double[][] b = {{7, 8, 9, 10}, {11, 12, 13, 14}, {15, 16, 17, 18}};
                double[][] c = {{74, 80, 86, 92}, {173, 188, 203, 218}};
                return allNumericDataTypeVariants(a, b, c);
            }

            @ParameterizedTest
            @MethodSource("getMatmul_test_2x3by3x4Data")
            void matmul_test_2x3by3x4(final ITensor a, final ITensor b, ITensor c) {
                ITensor result = a.matmul(b);
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

            Stream<Arguments> getMatmul_test_withBTransposedAsExtraOpTensor() {
                double[][] a = {{1, 2, 3}, {4, 5, 6}};
                double[][] b = {{7, 11, 15}, {8, 12, 16}, {9, 13, 17}, {10, 14, 18}};
                double[][] c = {{74, 80, 86, 92}, {173, 188, 203, 218}};
                return allNumericDataTypeVariants(a, b, c).map(args -> {
                    ITensor aMatrix = (ITensor) args.get()[0];
                    ITensor bMatrix = (ITensor) args.get()[1];
                    ITensor cMatrix = (ITensor) args.get()[2];
                    return Arguments.of(aMatrix, bMatrix.transpose(), cMatrix);
                });
            }

            @Disabled // enable when strides are supported in matmul
            @ParameterizedTest
            @MethodSource("getMatmul_test_withBTransposedAsExtraOpTensor")
            void matmul_test_withBTransposedAsExtraOpTensor(final ITensor a, final ITensor b, ITensor c) {
                ITensor result = a.matmul(b);
                assertEquals(c, result);
            }

            Stream<Arguments> getMatmul_test_withATransposedAsExtraOpTensor() {
                double[][] a = {{1, 4}, {2, 5}, {3, 6}};
                double[][] b = {{7, 8, 9, 10}, {11, 12, 13, 14}, {15, 16, 17, 18}};
                double[][] c = {{74, 80, 86, 92}, {173, 188, 203, 218}};
                return allNumericDataTypeVariants(a, b, c).map(args -> {
                    ITensor aMatrix = (ITensor) args.get()[0];
                    ITensor bMatrix = (ITensor) args.get()[1];
                    ITensor cMatrix = (ITensor) args.get()[2];
                    return Arguments.of(aMatrix.transpose(), bMatrix, cMatrix);
                });
            }

            @Disabled // enable when strides are supported in matmul
            @ParameterizedTest
            @MethodSource("getMatmul_test_withATransposedAsExtraOpTensor")
            void matmul_test_withATransposedAsExtraOpTensor(final ITensor a, final ITensor b, ITensor c) {
                ITensor result = a.matmul(b);
                assertEquals(c, result);
            }

            Stream<Arguments> getMatmul_test_withATransposedAsExtraOpTensor_and_BTransposedAsExtraOpTensor() {
                double[][] a = {{1, 4}, {2, 5}, {3, 6}};
                double[][] b = {{7, 11, 15}, {8, 12, 16}, {9, 13, 17}, {10, 14, 18}};
                double[][] c = {{74, 80, 86, 92}, {173, 188, 203, 218}};
                return allNumericDataTypeVariants(a, b, c).map(args -> {
                    ITensor aMatrix = (ITensor) args.get()[0];
                    ITensor bMatrix = (ITensor) args.get()[1];
                    ITensor cMatrix = (ITensor) args.get()[2];
                    return Arguments.of(aMatrix.transpose(), bMatrix.transpose(), cMatrix);
                });
            }

            @Disabled // enable when strides are supported in matmul
            @ParameterizedTest
            @MethodSource("getMatmul_test_withATransposedAsExtraOpTensor_and_BTransposedAsExtraOpTensor")
            void matmul_test_withATransposedAsExtraOpTensor_and_BTransposedAsExtraOpTensor(final ITensor a, final ITensor b, ITensor c) {
                ITensor result = a.matmul(b);
                assertEquals(c, result);
            }

            Stream<Arguments> getMatmul_test_withJvmTensorData() {
                SciCore jvmSciCore = new SciCore();
                jvmSciCore.addBackend(ISciCore.BackendType.JVM);
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
        @TestInstance(TestInstance.Lifecycle.PER_CLASS)
        class Matmul3d {

            Stream<Arguments> getNumericDataTypes() {
                return Arrays.stream(DataType.values())
                        .filter(DataType::isNumeric)
                        .map(Arguments::of);
            }

            @ParameterizedTest
            @MethodSource("getNumericDataTypes")
            void matmul_test_3dby3d(final DataType dataType) {
                ITensor a = sciCore.arange(0, 24, 1, dataType).view(2, 3, 4);
                ITensor b = sciCore.arange(0, 32, 1, dataType).view(2, 4, 4);
                ITensor c = a.matmul(b);
                ITensor expected = sciCore.ndarray(new float[][][]{{{56, 62, 68, 74}, {152, 174, 196, 218}, {248, 286, 324, 362}},
                                {{1208, 1262, 1316, 1370}, {1560, 1630, 1700, 1770}, {1912, 1998, 2084, 2170}}})
                        .cast(dataType);
                assertEquals(expected, c);
            }

            @ParameterizedTest
            @MethodSource("getNumericDataTypes")
            void matmul_test_2dby3d(final DataType dataType) {
                ITensor a = sciCore.arange(0, 12, 1, dataType).view(2, 6);
                ITensor b = sciCore.arange(0, 24, 1, dataType).view(2, 6, 2);
                ITensor c = a.matmul(b);
                ITensor expected = sciCore
                        .ndarray(new float[][][]{{{110, 125}, {290, 341}}, {{290, 305}, {902, 953}}})
                        .cast(dataType);
                assertEquals(expected, c);
            }

            @ParameterizedTest
            @MethodSource("getNumericDataTypes")
            void matmul_test_2dby3d_2(final DataType dataType) {
                ITensor a = sciCore.arange(0, 12, 1, dataType).view(2, 6);
                ITensor b = sciCore.arange(0, 36, 1, dataType).view(2, 6, 3);
                ITensor c = a.matmul(b);
                ITensor expected = sciCore
                        .ndarray(new float[][][]{{{165, 180, 195}, {435, 486, 537}}, {{435, 450, 465}, {1353, 1404, 1455}}})
                        .cast(dataType);
                assertEquals(expected, c);
            }

            @ParameterizedTest
            @MethodSource("getNumericDataTypes")
            void matmul_test_3dby2d(final DataType dataType) {
                ITensor a = sciCore.arange(0, 36, 1, dataType).view(2, 6, 3);
                ITensor b = sciCore.arange(0, 12, 1, dataType).view(3, 4);
                ITensor c = a.matmul(b);
                ITensor expected = sciCore
                        .ndarray(new float[][][]{
                                {
                                        {20, 23, 26, 29}, {56, 68, 80, 92}, {92, 113, 134, 155},
                                        {128, 158, 188, 218}, {164, 203, 242, 281}, {200, 248, 296, 344}},
                                {
                                        {236, 293, 350, 407}, {272, 338, 404, 470}, {308, 383, 458, 533},
                                        {344, 428, 512, 596}, {380, 473, 566, 659}, {416, 518, 620, 722}
                                }
                        })
                        .cast(dataType);
                assertEquals(expected, c);
            }

            @ParameterizedTest
            @MethodSource("getNumericDataTypes")
            void matmul_test_3dby3d_firstBatchSizeIsOne(final DataType dataType) {
                ITensor a = sciCore.arange(0, 15, 1, dataType).view(1, 3, 5);
                ITensor b = sciCore.arange(0, 10, 1, dataType).view(2, 5, 1);
                ITensor c = a.matmul(b);
                ITensor expected = sciCore
                        .ndarray(new float[][][]{{{30}, {80}, {130}}, {{80}, {255}, {430}}})
                        .cast(dataType);
                assertEquals(expected, c);
            }
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

    @Nested
    class LeftDivide {

        @Test
        void left_divide_test_1by2x2x2() {
            ITensor a = sciCore.ndarray(new float[][][]{{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}});
            ITensor b = sciCore.scalar(1.0f);
            ITensor result = a.leftDivide(b);
            assertEquals(sciCore.ndarray(new float[][][]{{{1.0f, 1.0f / 2.0f}, {1.0f / 3.0f, 1.0f / 4.0f}}, {{1.0f / 5.0f, 1.0f / 6.0f}, {1.0f / 7.0f, 1.0f / 8.0f}}}), result);
        }

    }

    @Nested
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
        void plus_test_2x3byScalar_success() {
            ITensor a = sciCore.matrix(new float[][]{{1, 2, 3}, {4, 5, 6}});
            ITensor result = a.plus(10f);
            assertEquals(sciCore.matrix(new float[][]{{11, 12, 13}, {14, 15, 16}}), result);
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
    class PlusInplaceTest {

        @Test
        void plusInplace_test_1x1by1x1_success() {
            ITensor a = sciCore.scalar(1.0f);
            ITensor b = sciCore.scalar(2.0f);
            a.add(b);
            assertEquals(sciCore.scalar(3.0f), a);
        }

        @Test
        void plusInplace_test_2x3by2x3_success() {
            ITensor a = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
            ITensor b = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
            a.add(b);
            assertEquals(sciCore.matrix(new float[][]{{2.0f, 4.0f, 6.0f}, {8.0f, 10.0f, 12.0f}}), a);
        }

        @Test
        void plusInplace_test_2x3byScalar_success() {
            ITensor a = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
            ITensor b = sciCore.scalar(2.0f);
            a.add(b);
            assertEquals(sciCore.matrix(new float[][]{{3.0f, 4.0f, 5.0f}, {6.0f, 7.0f, 8.0f}}), a);
        }

        @Test
        void plusInplace_test_2x3by3x3_broadcastFirstDim_broadcastImpossible_failure() {
            // (2, 3) + (3, 3) = (2, 3)
            ITensor a = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
            ITensor b = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}});
            assertThrows(IllegalArgumentException.class, () -> a.add(b));
        }

        @Test
        void plusInplace_test_2x3by1x3_broadcastSecondDim_success() {
            // (2, 3) + (1, 3) = (2, 3)
            ITensor a = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
            ITensor b = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}});
            a.add(b);
            assertEquals(sciCore.matrix(new float[][]{{2.0f, 4.0f, 6.0f}, {5.0f, 7.0f, 9.0f}}), a);
        }

        @Test
        void plusInplace_test_1x3by2x3_broadcastSecondDim_rhsShapeGreaterThanLhsShape_failure() {
            // (1, 3) + (2, 3)
            ITensor a = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}});
            ITensor b = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
            assertThrows(IllegalArgumentException.class, () -> a.add(b));
        }

        @Test
        void plusInplace_test_2x3x4by2x3x4_success() {
            // (2, 3, 4) + (2, 3, 4)
            ITensor a = sciCore.ndarray(new float[][][]{
                    {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}},
                    {{13.0f, 14.0f, 15.0f, 16.0f}, {17.0f, 18.0f, 19.0f, 20.0f}, {21.0f, 22.0f, 23.0f, 24.0f}}});
            ITensor b = sciCore.ndarray(new float[][][]{
                    {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}},
                    {{13.0f, 14.0f, 15.0f, 16.0f}, {17.0f, 18.0f, 19.0f, 20.0f}, {21.0f, 22.0f, 23.0f, 24.0f}}});
            a.add(b);
            assertEquals(sciCore.ndarray(new float[][][]{
                    {{2.0f, 4.0f, 6.0f, 8.0f}, {10.0f, 12.0f, 14.0f, 16.0f}, {18.0f, 20.0f, 22.0f, 24.0f}},
                    {{26.0f, 28.0f, 30.0f, 32.0f}, {34.0f, 36.0f, 38.0f, 40.0f}, {42.0f, 44.0f, 46.0f, 48.0f}}}), a);
        }

        @Test
        void plusInplace_test_2x3x4by3x4_success() {
            // (2, 3, 4) + (3, 4)
            ITensor a = sciCore.ndarray(new float[][][]{
                    {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}},
                    {{13.0f, 14.0f, 15.0f, 16.0f}, {17.0f, 18.0f, 19.0f, 20.0f}, {21.0f, 22.0f, 23.0f, 24.0f}}});
            ITensor b = sciCore.ndarray(new float[][]{{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}});
            a.add(b);
            assertEquals(sciCore.ndarray(new float[][][]{
                    {{2.0f, 4.0f, 6.0f, 8.0f}, {10.0f, 12.0f, 14.0f, 16.0f}, {18.0f, 20.0f, 22.0f, 24.0f}},
                    {{14.0f, 16.0f, 18.0f, 20.0f}, {22.0f, 24.0f, 26.0f, 28.0f}, {30.0f, 32.0f, 34.0f, 36.0f}}}), a);
        }

        @Test
        void plusInplace_test_2x3x4by4_success() {
            // (2, 3, 4) + (4)
            ITensor a = sciCore.ndarray(new float[][][]{
                    {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}},
                    {{13.0f, 14.0f, 15.0f, 16.0f}, {17.0f, 18.0f, 19.0f, 20.0f}, {21.0f, 22.0f, 23.0f, 24.0f}}});
            ITensor b = sciCore.ndarray(new float[]{1.0f, 2.0f, 3.0f, 4.0f});
            a.add(b);
            assertEquals(sciCore.ndarray(new float[][][]{
                    {{2.0f, 4.0f, 6.0f, 8.0f}, {6.0f, 8.0f, 10.0f, 12.0f}, {10.0f, 12.0f, 14.0f, 16.0f}},
                    {{14.0f, 16.0f, 18.0f, 20.0f}, {18.0f, 20.0f, 22.0f, 24.0f}, {22.0f, 24.0f, 26.0f, 28.0f}}}), a);
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
        void minus_test_2x3byScalar_success() {
            ITensor a = sciCore.matrix(new float[][]{{1, 2, 3}, {4, 5, 6}});
            ITensor b = sciCore.scalar(2.0f);
            ITensor result = a.minus(b);
            assertEquals(sciCore.matrix(new float[][]{{-1, 0, 1}, {2, 3, 4}}), result);
        }

        @Test
        void minus_test_2x3by1x3_success() {
            ITensor a = sciCore.matrix(new float[][]{{1, 2, 3}, {4, 5, 6}});
            ITensor b = sciCore.matrix(new float[][]{{1, 2, 3}});
            ITensor result = a.minus(b);
            assertEquals(sciCore.matrix(new float[][]{{0, 0, 0}, {3, 3, 3}}), result);
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
    class MinusInplaceTest {

        @Test
        void minusInplace_test_1x1by1x1_success() {
            ITensor a = sciCore.matrix(new float[][]{{1.0f}});
            ITensor b = sciCore.matrix(new float[][]{{2.0f}});
            a.subtract(b);
            assertEquals(sciCore.matrix(new float[][]{{-1.0f}}), a);
        }

        @Test
        void minusInplace_test_2x2by2x2_success() {
            ITensor a = sciCore.matrix(new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}});
            ITensor b = sciCore.matrix(new float[][]{{5.0f, 6.0f}, {7.0f, 8.0f}});
            a.subtract(b);
            assertEquals(sciCore.matrix(new float[][]{{-4.0f, -4.0f}, {-4.0f, -4.0f}}), a);
        }

        @Test
        void minusInplace_test_2x2by2x1_success() {
            ITensor a = sciCore.matrix(new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}});
            ITensor b = sciCore.matrix(new float[][]{{5.0f}, {6.0f}});
            a.subtract(b);
            assertEquals(sciCore.matrix(new float[][]{{-4.0f, -3.0f}, {-3.0f, -2.0f}}), a);
        }

        @Test
        void minusInplace_test_2x3ByScalar_success() {
            ITensor a = sciCore.matrix(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
            ITensor b = sciCore.scalar(2.0f);
            a.subtract(b);
            assertEquals(sciCore.matrix(new float[][]{{-1.0f, 0.0f, 1.0f}, {2.0f, 3.0f, 4.0f}}), a);
        }

        @Test
        void minusInplace_test_2x3x4by2x3x4_success() {
            // (2, 3, 4) - (2, 3, 4) = (2, 3, 4)
            ITensor a = sciCore.ndarray(new float[][][]{
                    {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}},
                    {{13.0f, 14.0f, 15.0f, 16.0f}, {17.0f, 18.0f, 19.0f, 20.0f}, {21.0f, 22.0f, 23.0f, 24.0f}}});
            ITensor b = sciCore.ndarray(new float[][][]{
                    {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}},
                    {{13.0f, 14.0f, 15.0f, 16.0f}, {17.0f, 18.0f, 19.0f, 20.0f}, {21.0f, 22.0f, 23.0f, 24.0f}}});
            a.subtract(b);
            assertEquals(sciCore.ndarray(new float[][][]{
                    {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}},
                    {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}}}), a);
        }

        @Test
        void minusInplace_test_1x3x4by2x3x4_rhsShapeGreaterThanLhsShape_failure() {
            // (1, 3, 4) - (2, 3, 4)
            ITensor a = sciCore.ndarray(new float[][][]{
                    {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}},
                    {{13.0f, 14.0f, 15.0f, 16.0f}, {17.0f, 18.0f, 19.0f, 20.0f}, {21.0f, 22.0f, 23.0f, 24.0f}}});
            ITensor b = sciCore.ndarray(new float[][][]{
                    {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}},
                    {{13.0f, 14.0f, 15.0f, 16.0f}, {17.0f, 18.0f, 19.0f, 20.0f}, {21.0f, 22.0f, 23.0f, 24.0f}},
                    {{25.0f, 26.0f, 27.0f, 28.0f}, {29.0f, 30.0f, 31.0f, 32.0f}, {33.0f, 34.0f, 35.0f, 36.0f}}});
            assertThrows(IllegalArgumentException.class, () -> a.subtract(b));
        }

        @Test
        void minusInplace_test_2x3x4by2x3x2_failure() {
            // (2, 3, 4) - (2, 3, 2)
            ITensor a = sciCore.ndarray(new float[][][]{
                    {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}},
                    {{13.0f, 14.0f, 15.0f, 16.0f}, {17.0f, 18.0f, 19.0f, 20.0f}, {21.0f, 22.0f, 23.0f, 24.0f}}});
            ITensor b = sciCore.ndarray(new float[][][]{
                    {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
                    {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}},
            });
            assertThrows(IllegalArgumentException.class, () -> a.subtract(b));
        }

        @Test
        void minusInplace_test_2x3x4by2x3x1_success() {
            // (2, 3, 4) - (2, 3, 1)
            ITensor a = sciCore.ndarray(new float[][][]{
                    {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}},
                    {{13.0f, 14.0f, 15.0f, 16.0f}, {17.0f, 18.0f, 19.0f, 20.0f}, {21.0f, 22.0f, 23.0f, 24.0f}}});
            ITensor b = sciCore.ndarray(new float[][][]{
                    {{1.0f}, {2.0f}, {3.0f}},
                    {{4.0f}, {5.0f}, {6.0f}},
            });
            a.subtract(b);
            assertEquals(sciCore.ndarray(new float[][][]{
                    {{0.0f, 1.0f, 2.0f, 3.0f}, {3.0f, 4.0f, 5.0f, 6.0f}, {6.0f, 7.0f, 8.0f, 9.0f}},
                    {{9.0f, 10.0f, 11.0f, 12.0f}, {12.0f, 13.0f, 14.0f, 15.0f}, {15.0f, 16.0f, 17.0f, 18.0f}}}), a);
        }

        @Test
        void minusInplace_test_2x3x4by2x1x4_success() {
            // (2, 3, 4) - (2, 1, 4)
            ITensor a = sciCore.ndarray(new float[][][]{
                    {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}},
                    {{13.0f, 14.0f, 15.0f, 16.0f}, {17.0f, 18.0f, 19.0f, 20.0f}, {21.0f, 22.0f, 23.0f, 24.0f}}});
            ITensor b = sciCore.ndarray(new float[][][]{
                    {{1.0f, 2.0f, 3.0f, 4.0f}},
                    {{5.0f, 6.0f, 7.0f, 8.0f}},
            });
            a.subtract(b);
            assertEquals(sciCore.ndarray(new float[][][]{
                    {{0.0f, 0.0f, 0.0f, 0.0f}, {4.0f, 4.0f, 4.0f, 4.0f}, {8.0f, 8.0f, 8.0f, 8.0f}},
                    {{8.0f, 8.0f, 8.0f, 8.0f}, {12.0f, 12.0f, 12.0f, 12.0f}, {16.0f, 16.0f, 16.0f, 16.0f}}}), a);
        }

        @Test
        void minusInplace_test_2x3x4by2x1x1_success() {
            // (2, 3, 4) - (2, 1, 1)
            ITensor a = sciCore.ndarray(new float[][][]{
                    {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}},
                    {{13.0f, 14.0f, 15.0f, 16.0f}, {17.0f, 18.0f, 19.0f, 20.0f}, {21.0f, 22.0f, 23.0f, 24.0f}}});
            ITensor b = sciCore.ndarray(new float[][][]{
                    {{1.0f}},
                    {{2.0f}},
            });
            a.subtract(b);
            assertEquals(sciCore.ndarray(new float[][][]{
                    {{0.0f, 1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f, 7.0f}, {8.0f, 9.0f, 10.0f, 11.0f}},
                    {{11.0f, 12.0f, 13.0f, 14.0f}, {15.0f, 16.0f, 17.0f, 18.0f}, {19.0f, 20.0f, 21.0f, 22.0f}}}), a);
        }

        @Test
        void minusInplace_test_2x3x4by1x3x4_success() {
            // (2, 3, 4) - (1, 3, 4)
            ITensor a = sciCore.ndarray(new float[][][]{
                    {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}},
                    {{13.0f, 14.0f, 15.0f, 16.0f}, {17.0f, 18.0f, 19.0f, 20.0f}, {21.0f, 22.0f, 23.0f, 24.0f}}});
            ITensor b = sciCore.ndarray(new float[][][]{
                    {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}},
            });
            a.subtract(b);
            assertEquals(sciCore.ndarray(new float[][][]{
                    {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}},
                    {{12.0f, 12.0f, 12.0f, 12.0f}, {12.0f, 12.0f, 12.0f, 12.0f}, {12.0f, 12.0f, 12.0f, 12.0f}}}), a);
        }

        @Test
        void minusInplace_test_2x3x4by1x3x1_success() {
            // (2, 3, 4) - (1, 3, 1)
            ITensor a = sciCore.ndarray(new float[][][]{
                    {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}},
                    {{13.0f, 14.0f, 15.0f, 16.0f}, {17.0f, 18.0f, 19.0f, 20.0f}, {21.0f, 22.0f, 23.0f, 24.0f}}});
            ITensor b = sciCore.ndarray(new float[][][]{
                    {{1.0f}, {5.0f}, {9.0f}},
            });
            a.subtract(b);
            assertEquals(sciCore.ndarray(new float[][][]{
                    {{0.0f, 1.0f, 2.0f, 3.0f}, {0.0f, 1.0f, 2.0f, 3.0f}, {0.0f, 1.0f, 2.0f, 3.0f}},
                    {{12.0f, 13.0f, 14.0f, 15.0f}, {12.0f, 13.0f, 14.0f, 15.0f}, {12.0f, 13.0f, 14.0f, 15.0f}}}), a);
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
        void multiply_test_2x3ByScalar_success() {
            ITensor a = sciCore.matrix(new float[][]{{2, 3, 4}, {5, 6, 7}});
            ITensor b = sciCore.scalar(10f);

            ITensor c = a.multiply(b);
            assertEquals(sciCore.matrix(new float[][]{{20.0f, 30.0f, 40.0f}, {50.0f, 60.0f, 70.0f}}), c);
        }

        @Test
        void multiply_test_scalarByScalar_success() {
            ITensor a = sciCore.scalar(2f);
            ITensor b = sciCore.scalar(10f);

            ITensor c = a.multiply(b);
            assertEquals(sciCore.scalar(20.0f), c);
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
        void multiply_test_2by2x1x1_broadcast_success() {
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

        @Test
        void test_argmax_3d_dimMinus1() {
            ITensor a = sciCore.ndarray(new float[][][]{
                    {{2, 1, 0}, {1, 0, 3}},
                    {{1, 0, 3}, {2, 1, 0}}
            });
            ITensor max2 = a.argmax(-1);
            assertEquals(sciCore.scalar(5L), max2);
        }

        @Test
        void test_argmax_3d_dim0() {
            ITensor a = sciCore.ndarray(new float[][][]{
                    {{2, 1, 0}, {1, 0, 3}},
                    {{1, 0, 3}, {2, 1, 0}}
            });
            ITensor max1 = a.argmax(0);
            assertEquals(sciCore.ndarray(new long[][]{{0, 0, 1}, {1, 1, 0}}), max1);
        }

        @Test
        void test_argmax_3d_dim1() {
            ITensor a = sciCore.ndarray(new float[][][]{
                    {{2, 1, 0}, {1, 0, 3}},
                    {{1, 0, 3}, {2, 1, 0}}
            });
            ITensor max2 = a.argmax(1);
            assertEquals(sciCore.ndarray(new long[][]{
                    {0, 0, 1},
                    {1, 1, 0}
            }), max2);
        }

        @Test
        void test_argmax_3d_dim2() {
            ITensor a = sciCore.ndarray(new float[][][]{
                    {{2, 1, 0}, {1, 0, 3}},
                    {{1, 0, 3}, {2, 1, 0}}
            });
            ITensor max2 = a.argmax(2);
            assertEquals(sciCore.ndarray(new long[][]{
                    {0, 2},
                    {2, 0}
            }), max2);
        }

        @Test
        void test_argmax_4d_dimMinus1() {
            ITensor a = sciCore.ndarray(new float[][][][]{
                    {{{0, 4, 1, 3}, {4, 3, 7, 1}}, {{3, 4, 1, 9}, {5, 1, 0, 8}}, {{4, 5, 9, 4}, {1, 1, 6, 4}}},
                    {{{7, 1, 6, 3}, {1, 6, 6, 6}}, {{5, 0, 2, 6}, {4, 1, 5, 0}}, {{6, 8, 5, 7}, {5, 1, 5, 8}}}
            });
            ITensor max2 = a.argmax(-1);
            assertEquals(sciCore.scalar(11L), max2);
        }

        @Test
        void test_argmax_4d_dim0() {
            ITensor a = sciCore.ndarray(new float[][][][]{
                    {{{0, 4, 1, 3}, {4, 3, 7, 1}}, {{3, 4, 1, 9}, {5, 1, 0, 8}}, {{4, 5, 9, 4}, {1, 1, 6, 4}}},
                    {{{7, 1, 6, 3}, {1, 6, 6, 6}}, {{5, 0, 2, 6}, {4, 1, 5, 0}}, {{6, 8, 5, 7}, {5, 1, 5, 8}}}
            });
            ITensor max1 = a.argmax(0);
            assertEquals(sciCore.ndarray(new long[][][]{
                    {{1, 0, 1, 0}, {0, 1, 0, 1}},
                    {{1, 0, 1, 0}, {0, 0, 1, 0}},
                    {{1, 1, 0, 1}, {1, 0, 0, 1}}
            }), max1);
        }

        @Test
        void test_argmax_4d_dim1() {
            ITensor a = sciCore.ndarray(new float[][][][]{
                    {{{0, 4, 1, 3}, {4, 3, 7, 1}}, {{3, 4, 1, 9}, {5, 1, 0, 8}}, {{4, 5, 9, 4}, {1, 1, 6, 4}}},
                    {{{7, 1, 6, 3}, {1, 6, 6, 6}}, {{5, 0, 2, 6}, {4, 1, 5, 0}}, {{6, 8, 5, 7}, {5, 1, 5, 8}}}
            });
            ITensor max1 = a.argmax(1);
            assertEquals(sciCore.ndarray(new long[][][]{
                    {{2, 2, 2, 1}, {1, 0, 0, 1}},
                    {{0, 2, 0, 2}, {2, 0, 0, 2}}
            }), max1);
        }

        @Test
        void test_argmax_4d_dim2() {
            ITensor a = sciCore.ndarray(new float[][][][]{
                    {{{0, 4, 1, 3}, {4, 3, 7, 1}}, {{3, 4, 1, 9}, {5, 1, 0, 8}}, {{4, 5, 9, 4}, {1, 1, 6, 4}}},
                    {{{7, 1, 6, 3}, {1, 6, 6, 6}}, {{5, 0, 2, 6}, {4, 1, 5, 0}}, {{6, 8, 5, 7}, {5, 1, 5, 8}}}
            });
            ITensor max1 = a.argmax(2);
            assertEquals(sciCore.ndarray(new long[][][]{
                    {{1, 0, 1, 0}, {1, 0, 0, 0}, {0, 0, 0, 0}},
                    {{0, 1, 0, 1}, {0, 1, 1, 0}, {0, 0, 0, 1}}
            }), max1);
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
    }

    @Nested
    class InplaceOperationChains {

        @Test
        void test_two_inplace_operations_chain() {
            ITensor a = sciCore.matrix(new float[][]{
                    {1, 5, 3}, {6, 3, 2}
            });
            ITensor b = sciCore.matrix(new float[][]{
                    {3, 2, 5}, {2, 4, 6}
            });
            ITensor c = sciCore.matrix(new float[][]{
                    {9, 2, 4}, {5, 2, 9}
            });
            a.add(b);
            a.add(c);
            assertEquals(sciCore.matrix(new float[][]{
                    {13, 9, 12}, {13, 9, 17}
            }), a);
        }

        @Test
        void test_three_inplace_operations_chain() {
            ITensor a = sciCore.matrix(new float[][]{
                    {1, 5, 3}, {6, 3, 2}
            });
            ITensor b = sciCore.matrix(new float[][]{
                    {3, 2, 5}, {2, 4, 6}
            });
            ITensor c = sciCore.matrix(new float[][]{
                    {9, 2, 4}, {5, 2, 9}
            });
            ITensor d = sciCore.matrix(new float[][]{
                    {1, 2, 3}, {4, 5, 6}
            });
            a.add(b);
            a.add(c);
            a.add(d);
            assertEquals(sciCore.matrix(new float[][]{
                    {14, 11, 15}, {17, 14, 23}
            }), a);
        }

        @Test
        void test_four_inplace_operations_chain() {
            ITensor a = sciCore.matrix(new float[][]{
                    {1, 5, 3}, {6, 3, 2}
            });
            ITensor b = sciCore.matrix(new float[][]{
                    {3, 2, 5}, {2, 4, 6}
            });
            ITensor c = sciCore.matrix(new float[][]{
                    {9, 2, 4}, {5, 2, 9}
            });
            ITensor d = sciCore.matrix(new float[][]{
                    {1, 2, 3}, {4, 5, 6}
            });
            ITensor e = sciCore.matrix(new float[][]{
                    {3, 2, 6}, {3, 1, 2}
            });
            a.add(b);
            a.add(c);
            a.add(d);
            a.add(e);
            assertEquals(sciCore.matrix(new float[][]{
                    {17, 13, 21}, {20, 15, 25}
            }), a);
        }


        @Test
        void test_inplace_after_op_without_result() {
            ITensor a = sciCore.matrix(new float[][]{
                    {1, 5, 3}, {6, 3, 2}
            });
            ITensor b = sciCore.matrix(new float[][]{
                    {3, 2, 5}, {2, 4, 6}
            });
            ITensor c = sciCore.matrix(new float[][]{
                    {9, 2, 4}, {5, 2, 9}
            });
            ITensor d = a.multiply(b);
            d.add(c);
            assertEquals(sciCore.matrix(new float[][]{
                    {12, 12, 19}, {17, 14, 21}
            }), d);
        }

        @Test
        void test_inplace_after_op_with_result() {
            ITensor a = sciCore.matrix(new float[][]{
                    {1, 5, 3}, {6, 3, 2}
            });
            ITensor b = sciCore.matrix(new float[][]{
                    {3, 2, 5}, {2, 4, 6}
            });
            ITensor c = sciCore.matrix(new float[][]{
                    {9, 2, 4}, {5, 2, 9}
            });
            ITensor d = a.multiply(b);
            d.getAsFloatFlat(0); // force computation
            d.add(c);
            assertEquals(sciCore.matrix(new float[][]{
                    {12, 12, 19}, {17, 14, 21}
            }), d);
        }

        @Test
        void test_multiple_inplace_with_same_second_input_tensor_after_op_without_result() {
            ITensor a = sciCore.matrix(new float[][]{
                    {1, 5, 3}, {6, 3, 2}
            });
            ITensor b = sciCore.matrix(new float[][]{
                    {3, 2, 5}, {2, 4, 6}
            });
            ITensor c = sciCore.matrix(new float[][]{
                    {9, 2, 4}, {5, 2, 9}
            });
            ITensor d = a.multiply(b);
            d.add(c);
            d.add(c);
            assertEquals(sciCore.matrix(new float[][]{
                    {21, 14, 23}, {22, 16, 30}
            }), d);
        }

        @Test
        void test_multiple_inplace_with_same_second_input_tensor_after_op_with_result() {
            ITensor a = sciCore.matrix(new float[][]{
                    {1, 5, 3}, {6, 3, 2}
            });
            ITensor b = sciCore.matrix(new float[][]{
                    {3, 2, 5}, {2, 4, 6}
            });
            ITensor c = sciCore.matrix(new float[][]{
                    {9, 2, 4}, {5, 2, 9}
            });
            ITensor d = a.multiply(b);
            d.getAsFloatFlat(0); // force computation
            d.add(c);
            d.add(c);
            assertEquals(sciCore.matrix(new float[][]{
                    {21, 14, 23}, {22, 16, 30}
            }), d);
        }
    }


    @Nested
    class Get {

        @Test
        void test_get_1dIdxInto2d_success() {
            ITensor a = sciCore.matrix(new float[][]{
                    {1, 5, 3}, {6, 3, 2}, {8, 9, 1}, {5, 2, 3}
            });
            ITensor idx = sciCore.array(new int[]{1, 1, 3, 2, 2});
            ITensor b = a.get(idx);
            assertEquals(sciCore.matrix(new float[][]{
                    {6, 3, 2}, {6, 3, 2}, {5, 2, 3}, {8, 9, 1}, {8, 9, 1}
            }), b);
        }

        @Test
        void test_get_2dIdxInto2d_success() {
            ITensor a = sciCore.matrix(new float[][]{
                    {1, 5, 3}, {6, 3, 2}, {8, 9, 1}, {5, 2, 3}
            });
            ITensor idx = sciCore.matrix(new int[][]{
                    {1, 1, 3, 2, 2}, {0, 1, 2, 0, 1}
            });
            ITensor b = a.get(idx);
            assertEquals(sciCore.ndarray(new float[][][]{
                    {{6, 3, 2}, {6, 3, 2}, {5, 2, 3}, {8, 9, 1}, {8, 9, 1}},
                    {{1, 5, 3}, {6, 3, 2}, {8, 9, 1}, {1, 5, 3}, {6, 3, 2}}
            }), b);
        }

        @Test
        void test_get2IndexTensorsInto2d_success() {
            ITensor a = sciCore.matrix(new float[][]{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}});
            ITensor b = a.get(sciCore.array(new long[]{0}), sciCore.array(new long[]{1}));
            ITensor c = a.get(sciCore.array(new long[]{1}), sciCore.array(new long[]{1}));
            ITensor d = a.get(sciCore.array(new long[]{0, 1}), sciCore.array(new long[]{1, 1}));
            assertEquals(sciCore.array(new float[]{2f}), b);
            assertEquals(sciCore.array(new float[]{7f}), c);
            assertEquals(sciCore.array(new float[]{2, 7}), d);
        }

        @Test
        void test_get3dTooManyIndices_failure() {
            ITensor a = sciCore.ndarray(new float[][][]{{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}});
            assertThrows(IllegalArgumentException.class, () -> a.get(
                    sciCore.array(new long[]{0, 0, 0}),
                    sciCore.array(new long[]{0, 0, 1}),
                    sciCore.array(new long[]{0, 1, 0}),
                    sciCore.array(new long[]{0, 1, 1}),
                    sciCore.array(new long[]{0, 0, 0})
            ));
        }

    }

    @Nested
    class Concat {

        @Test
        void test_concat_2dDim0_success() {
            ITensor a = sciCore.matrix(new float[][]{
                    {1}, {2}, {3}
            });
            ITensor b = sciCore.matrix(new float[][]{
                    {4}
            });
            ITensor c = a.concat(b, 0);
            assertEquals(sciCore.matrix(new float[][]{
                    {1}, {2}, {3}, {4}
            }), c);
        }

        @Test
        void test_concat_2dDim1_success() {
            ITensor a = sciCore.matrix(new float[][]{
                    {1, 2, 3}, {4, 5, 6}
            });
            ITensor b = sciCore.matrix(new float[][]{
                    {7, 8, 9}, {10, 11, 12}
            });
            ITensor c = a.concat(b, 1);
            assertEquals(sciCore.matrix(new float[][]{
                    {1, 2, 3, 7, 8, 9}, {4, 5, 6, 10, 11, 12}
            }), c);
        }

        @Test
        void test_concat_3dDim0_success() {
            ITensor a = sciCore.ndarray(new float[][][]{
                    {{1, 2, 3}, {4, 5, 6}},
                    {{7, 8, 9}, {10, 11, 12}}
            });
            ITensor b = sciCore.ndarray(new float[][][]{
                    {{13, 14, 15}, {16, 17, 18}}
            });
            ITensor c = a.concat(b, 0);
            assertEquals(sciCore.ndarray(new float[][][]{
                    {{1, 2, 3}, {4, 5, 6}},
                    {{7, 8, 9}, {10, 11, 12}},
                    {{13, 14, 15}, {16, 17, 18}}
            }), c);
        }

        @Test
        void test_concat_3dDim1_success() {
            ITensor a = sciCore.ndarray(new float[][][]{
                    {{1, 2, 3}, {4, 5, 6}},
                    {{7, 8, 9}, {10, 11, 12}}
            });
            ITensor b = sciCore.ndarray(new float[][][]{
                    {{13, 14, 15}, {16, 17, 18}},
                    {{19, 20, 21}, {22, 23, 24}}
            });
            ITensor c = a.concat(b, 1);
            assertEquals(sciCore.ndarray(new float[][][]{
                    {{1, 2, 3}, {4, 5, 6}, {13, 14, 15}, {16, 17, 18}},
                    {{7, 8, 9}, {10, 11, 12}, {19, 20, 21}, {22, 23, 24}}
            }), c);
        }

        @Test
        void test_concat_3dDim2_success() {
            ITensor a = sciCore.ndarray(new float[][][]{
                    {{1, 2, 3}, {4, 5, 6}},
                    {{7, 8, 9}, {10, 11, 12}}
            });
            ITensor b = sciCore.ndarray(new float[][][]{
                    {{13, 14}, {16, 17}},
                    {{19, 20}, {22, 23}}
            });
            ITensor c = a.concat(b, 2);
            assertEquals(sciCore.ndarray(new float[][][]{
                    {{1, 2, 3, 13, 14}, {4, 5, 6, 16, 17}},
                    {{7, 8, 9, 19, 20}, {10, 11, 12, 22, 23}}
            }), c);
        }
    }

    @Nested
    class Stack {

        @Test
        void test_stack3dDim0() {
            ITensor a = sciCore.arange(0, 24, 1, DataType.FLOAT32).view(2, 6, 2);
            ITensor b = sciCore.arange(24, 48, 1, DataType.FLOAT32).view(2, 6, 2);
            ITensor c = sciCore.arange(48, 72, 1, DataType.FLOAT32).view(2, 6, 2);
            ITensor d = sciCore.stack(0, a, b, c);
            assertEquals(sciCore.ndarray(new float[][][][]{
                    {
                            {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}},
                            {{12, 13}, {14, 15}, {16, 17}, {18, 19}, {20, 21}, {22, 23}}
                    },
                    {
                            {{24, 25}, {26, 27}, {28, 29}, {30, 31}, {32, 33}, {34, 35}},
                            {{36, 37}, {38, 39}, {40, 41}, {42, 43}, {44, 45}, {46, 47}}
                    },
                    {
                            {{48, 49}, {50, 51}, {52, 53}, {54, 55}, {56, 57}, {58, 59}},
                            {{60, 61}, {62, 63}, {64, 65}, {66, 67}, {68, 69}, {70, 71}}
                    }
            }), d);
        }

        @Test
        void test_stack3dDim1() {
            ITensor a = sciCore.arange(0, 24, 1, DataType.FLOAT32).view(2, 6, 2);
            ITensor b = sciCore.arange(24, 48, 1, DataType.FLOAT32).view(2, 6, 2);
            ITensor c = sciCore.arange(48, 72, 1, DataType.FLOAT32).view(2, 6, 2);
            ITensor d = sciCore.stack(1, a, b, c);
            assertEquals(sciCore.ndarray(new float[][][][]{
                    {
                            {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}},
                            {{24, 25}, {26, 27}, {28, 29}, {30, 31}, {32, 33}, {34, 35}},
                            {{48, 49}, {50, 51}, {52, 53}, {54, 55}, {56, 57}, {58, 59}}
                    },
                    {
                            {{12, 13}, {14, 15}, {16, 17}, {18, 19}, {20, 21}, {22, 23}},
                            {{36, 37}, {38, 39}, {40, 41}, {42, 43}, {44, 45}, {46, 47}},
                            {{60, 61}, {62, 63}, {64, 65}, {66, 67}, {68, 69}, {70, 71}}
                    }
            }), d);
        }

        @Test
        void test_stack3dDim2() {
            ITensor a = sciCore.arange(0, 24, 1, DataType.FLOAT32).view(2, 6, 2);
            ITensor b = sciCore.arange(24, 48, 1, DataType.FLOAT32).view(2, 6, 2);
            ITensor c = sciCore.arange(48, 72, 1, DataType.FLOAT32).view(2, 6, 2);
            ITensor d = sciCore.stack(2, a, b, c);
            assertEquals(sciCore.ndarray(new float[][][][]{
                    {
                            {{0, 1}, {24, 25}, {48, 49}},
                            {{2, 3}, {26, 27}, {50, 51}},
                            {{4, 5}, {28, 29}, {52, 53}},
                            {{6, 7}, {30, 31}, {54, 55}},
                            {{8, 9}, {32, 33}, {56, 57}},
                            {{10, 11}, {34, 35}, {58, 59}}
                    },
                    {
                            {{12, 13}, {36, 37}, {60, 61}},
                            {{14, 15}, {38, 39}, {62, 63}},
                            {{16, 17}, {40, 41}, {64, 65}},
                            {{18, 19}, {42, 43}, {66, 67}},
                            {{20, 21}, {44, 45}, {68, 69}},
                            {{22, 23}, {46, 47}, {70, 71}}
                    }
            }), d);
        }
    }

}