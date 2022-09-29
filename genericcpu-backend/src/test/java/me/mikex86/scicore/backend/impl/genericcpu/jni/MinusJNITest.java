package me.mikex86.scicore.backend.impl.genericcpu.jni;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.memory.DirectMemoryManager;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import static java.lang.Math.max;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class MinusJNITest {

    private final DirectMemoryManager directMemoryManager = new DirectMemoryManager();

    static {
        new GenCPUBackend(); // Load the native library
    }

    @SuppressWarnings("ConstantConditions") // Keep the code explicit, even if it's a bit verbose
    @NotNull
    public Stream<Arguments> getMinusData(int nElementsA, int nElementsB) {
        List<Arguments> argumentsList = new ArrayList<>();

        for (DataType aType : DataType.values()) {
            if (!aType.isNumeric()) {
                continue;
            }
            for (DataType bType : DataType.values()) {
                if (!bType.isNumeric()) {
                    continue;
                }
                int nElementsC = max(nElementsA, nElementsB);
                DataType cDataType = DataType.getLarger(aType, bType);
                DirectMemoryHandle aMemory = directMemoryManager.calloc(nElementsA, aType);
                DirectMemoryHandle bMemory = directMemoryManager.calloc(nElementsB, bType);
                DirectMemoryHandle cMemory = directMemoryManager.calloc(nElementsC, cDataType);

                DirectMemoryHandle expectedResult = directMemoryManager.calloc(nElementsC, cDataType);

                ByteBuffer aBuffer = aMemory.asByteBuffer();
                ByteBuffer bBuffer = bMemory.asByteBuffer();
                ByteBuffer expectedResultBuffer = expectedResult.asByteBuffer();
                Random random = new Random(123);
                for (int i = 0; i < nElementsA; i++) {
                    switch (aType) {
                        case INT8 -> aBuffer.put((byte) random.nextInt());
                        case INT16 -> aBuffer.putShort((short) random.nextInt());
                        case INT32 -> aBuffer.putInt(random.nextInt());
                        case INT64 -> aBuffer.putLong(random.nextLong());
                        case FLOAT32 -> aBuffer.putFloat(random.nextFloat());
                        case FLOAT64 -> aBuffer.putDouble(random.nextDouble());
                    }
                }
                for (int i = 0; i < nElementsB; i++) {
                    switch (bType) {
                        case INT8 -> bBuffer.put((byte) random.nextInt());
                        case INT16 -> bBuffer.putShort((short) random.nextInt());
                        case INT32 -> bBuffer.putInt(random.nextInt());
                        case INT64 -> bBuffer.putLong(random.nextLong());
                        case FLOAT32 -> bBuffer.putFloat(random.nextFloat());
                        case FLOAT64 -> bBuffer.putDouble(random.nextDouble());
                    }
                }
                aBuffer.flip();
                bBuffer.flip();
                // calculate result
                for (int i = 0; i < nElementsC; i++) {
                    if (cDataType.isFloatingPoint()) {
                        double a;
                        {
                            switch (aType) {
                                case INT8 -> a = aBuffer.get(i % nElementsA);
                                case INT16 -> a = aBuffer.asShortBuffer().get(i % nElementsA);
                                case INT32 -> a = aBuffer.asIntBuffer().get(i % nElementsA);
                                case INT64 -> a = aBuffer.asLongBuffer().get(i % nElementsA);
                                case FLOAT32 -> a = aBuffer.asFloatBuffer().get(i % nElementsA);
                                case FLOAT64 -> a = aBuffer.asDoubleBuffer().get(i % nElementsA);
                                default -> throw new IllegalStateException("Unexpected value: " + aType);
                            }
                        }
                        double b;
                        {
                            switch (bType) {
                                case INT8 -> b = bBuffer.get(i % nElementsB);
                                case INT16 -> b = bBuffer.asShortBuffer().get(i % nElementsB);
                                case INT32 -> b = bBuffer.asIntBuffer().get(i % nElementsB);
                                case INT64 -> b = bBuffer.asLongBuffer().get(i % nElementsB);
                                case FLOAT32 -> b = bBuffer.asFloatBuffer().get(i % nElementsB);
                                case FLOAT64 -> b = bBuffer.asDoubleBuffer().get(i % nElementsB);
                                default -> throw new IllegalStateException("Unexpected value: " + bType);
                            }
                        }
                        double result = a - b;
                        switch (cDataType) {
                            case FLOAT32 -> expectedResultBuffer.putFloat((float) result);
                            case FLOAT64 -> expectedResultBuffer.putDouble(result);
                        }
                    } else {
                        long a, b;
                        {
                            switch (aType) {
                                case INT8 -> a = aBuffer.get(i % nElementsA);
                                case INT16 -> a = aBuffer.asShortBuffer().get(i % nElementsA);
                                case INT32 -> a = aBuffer.asIntBuffer().get(i % nElementsA);
                                case INT64 -> a = aBuffer.asLongBuffer().get(i % nElementsA);
                                default -> throw new IllegalStateException("Unexpected value: " + aType);
                            }
                        }
                        {
                            switch (bType) {
                                case INT8 -> b = bBuffer.get(i % nElementsB);
                                case INT16 -> b = bBuffer.asShortBuffer().get(i % nElementsB);
                                case INT32 -> b = bBuffer.asIntBuffer().get(i % nElementsB);
                                case INT64 -> b = bBuffer.asLongBuffer().get(i % nElementsB);
                                default -> throw new IllegalStateException("Unexpected value: " + bType);
                            }
                        }
                        long result;
                        if (aType == DataType.INT8 && bType == DataType.INT8) {
                            result = (byte) ((byte) a - (byte) b);
                        } else if (aType == DataType.INT8 && bType == DataType.INT16) {
                            result = (short) ((byte) a - (short) b);
                        } else if (aType == DataType.INT8 && bType == DataType.INT32) {
                            result = (int) ((byte) a - (int) b);
                        } else if (aType == DataType.INT8 && bType == DataType.INT64) {
                            result = (long) ((byte) a - (long) b);
                        } else if (aType == DataType.INT16 && bType == DataType.INT8) {
                            result = (short) ((short) a - (byte) b);
                        } else if (aType == DataType.INT16 && bType == DataType.INT16) {
                            result = (short) ((short) a - (short) b);
                        } else if (aType == DataType.INT16 && bType == DataType.INT32) {
                            result = (int) ((short) a - (int) b);
                        } else if (aType == DataType.INT16 && bType == DataType.INT64) {
                            result = (long) ((short) a - (long) b);
                        } else if (aType == DataType.INT32 && bType == DataType.INT8) {
                            result = (int) ((int) a - (byte) b);
                        } else if (aType == DataType.INT32 && bType == DataType.INT16) {
                            result = (int) ((int) a - (short) b);
                        } else if (aType == DataType.INT32 && bType == DataType.INT32) {
                            result = (int) ((int) a - (int) b);
                        } else if (aType == DataType.INT32 && bType == DataType.INT64) {
                            result = (long) ((int) a - (long) b);
                        } else if (aType == DataType.INT64 && bType == DataType.INT8) {
                            result = (long) ((long) a - (byte) b);
                        } else if (aType == DataType.INT64 && bType == DataType.INT16) {
                            result = (long) ((long) a - (short) b);
                        } else if (aType == DataType.INT64 && bType == DataType.INT32) {
                            result = (long) ((long) a - (int) b);
                        } else if (aType == DataType.INT64 && bType == DataType.INT64) {
                            result = (long) ((long) a - (long) b);
                        } else {
                            throw new IllegalStateException("Unexpected value: " + aType + " " + bType);
                        }
                        switch (cDataType) {
                            case INT8 -> expectedResultBuffer.put((byte) result);
                            case INT16 -> expectedResultBuffer.putShort((short) result);
                            case INT32 -> expectedResultBuffer.putInt((int) result);
                            case INT64 -> expectedResultBuffer.putLong(result);
                        }
                    }
                }
                expectedResultBuffer.flip();
                argumentsList.add(
                        Arguments.of(
                                aMemory,
                                aType,
                                nElementsA,
                                bMemory,
                                bType,
                                nElementsB,
                                cMemory,
                                nElementsC,
                                expectedResult,
                                cDataType
                        )
                );
            }
        }

        return argumentsList.stream();
    }

    @NotNull
    public Stream<Arguments> getMinusData_sameLengthNoBroadcast() {
        return getMinusData(100, 100);
    }

    @NotNull
    public Stream<Arguments> getMinusData_differentLengthAGreaterBWithBroadcast() {
        return getMinusData(100, 50);
    }

    @NotNull
    public Stream<Arguments> getMinusData_differentLengthBGreaterAWithBroadcast() {
        return getMinusData(50, 100);
    }

    @NotNull
    public Stream<Arguments> getMinusData_differentLengthBIsScalar() {
        return getMinusData(100, 1);
    }

    @NotNull
    public Stream<Arguments> getMinusData_differentLengthAIsScalar() {
        return getMinusData(1, 100);
    }

    @ParameterizedTest
    @MethodSource("getMinusData_sameLengthNoBroadcast")
    void minus_sameLengthNoBroadcast_success(DirectMemoryHandle a,
                                             DataType aDataType,
                                             long nElementsA,
                                             DirectMemoryHandle b,
                                             DataType bDataType,
                                             long nElementsB,
                                             DirectMemoryHandle c,
                                             long nElementsC,
                                             DirectMemoryHandle expectedResult,
                                             DataType cDataType) {
        MinusJNI.minus(a.getNativePtr(), aDataType, nElementsA, b.getNativePtr(), bDataType, nElementsB, c.getNativePtr(), nElementsC, cDataType);
        ByteBuffer cBuffer = c.asByteBuffer();
        ByteBuffer expectedResultBuffer = expectedResult.asByteBuffer();
        DataType outputDataType = DataType.getLarger(aDataType, bDataType);

        validateResult(outputDataType, cBuffer, expectedResultBuffer, nElementsC);
    }

    @ParameterizedTest
    @MethodSource("getMinusData_differentLengthAGreaterBWithBroadcast")
    void minus_differentLengthAGreaterB_success(DirectMemoryHandle a,
                                                DataType aDataType,
                                                long nElementsA,
                                                DirectMemoryHandle b,
                                                DataType bDataType,
                                                long nElementsB,
                                                DirectMemoryHandle c,
                                                long nElementsC,
                                                DirectMemoryHandle expectedResult,
                                                DataType cDataType) {
        MinusJNI.minus(a.getNativePtr(), aDataType, nElementsA, b.getNativePtr(), bDataType, nElementsB, c.getNativePtr(), nElementsC, cDataType);
        ByteBuffer cBuffer = c.asByteBuffer();
        ByteBuffer expectedResultBuffer = expectedResult.asByteBuffer();
        DataType outputDataType = DataType.getLarger(aDataType, bDataType);

        validateResult(outputDataType, cBuffer, expectedResultBuffer, nElementsC);
    }

    @ParameterizedTest
    @MethodSource("getMinusData_differentLengthBGreaterAWithBroadcast")
    void minus_differentLengthBGreaterA_success(DirectMemoryHandle a,
                                                DataType aDataType,
                                                long nElementsA,
                                                DirectMemoryHandle b,
                                                DataType bDataType,
                                                long nElementsB,
                                                DirectMemoryHandle c,
                                                long nElementsC,
                                                DirectMemoryHandle expectedResult,
                                                DataType cDataType) {
        MinusJNI.minus(a.getNativePtr(), aDataType, nElementsA, b.getNativePtr(), bDataType, nElementsB, c.getNativePtr(), nElementsC, cDataType);
        ByteBuffer cBuffer = c.asByteBuffer();
        ByteBuffer expectedResultBuffer = expectedResult.asByteBuffer();
        DataType outputDataType = DataType.getLarger(aDataType, bDataType);

        validateResult(outputDataType, cBuffer, expectedResultBuffer, nElementsC);
    }

    @ParameterizedTest
    @MethodSource("getMinusData_differentLengthBIsScalar")
    void minus_differentLengthBIsScalar_success(DirectMemoryHandle a,
                                                DataType aDataType,
                                                long nElementsA,
                                                DirectMemoryHandle b,
                                                DataType bDataType,
                                                long nElementsB,
                                                DirectMemoryHandle c,
                                                long nElementsC,
                                                DirectMemoryHandle expectedResult,
                                                DataType cDataType) {
        MinusJNI.minus(a.getNativePtr(), aDataType, nElementsA, b.getNativePtr(), bDataType, nElementsB, c.getNativePtr(), nElementsC, cDataType);
        ByteBuffer cBuffer = c.asByteBuffer();
        ByteBuffer expectedResultBuffer = expectedResult.asByteBuffer();
        DataType outputDataType = DataType.getLarger(aDataType, bDataType);

        validateResult(outputDataType, cBuffer, expectedResultBuffer, nElementsC);
    }

    @ParameterizedTest
    @MethodSource("getMinusData_differentLengthAIsScalar")
    void minus_differentLengthAIsScalar_success(DirectMemoryHandle a,
                                                DataType aDataType,
                                                long nElementsA,
                                                DirectMemoryHandle b,
                                                DataType bDataType,
                                                long nElementsB,
                                                DirectMemoryHandle c,
                                                long nElementsC,
                                                DirectMemoryHandle expectedResult,
                                                DataType cDataType) {
        MinusJNI.minus(a.getNativePtr(), aDataType, nElementsA, b.getNativePtr(), bDataType, nElementsB, c.getNativePtr(), nElementsC, cDataType);
        ByteBuffer cBuffer = c.asByteBuffer();
        ByteBuffer expectedResultBuffer = expectedResult.asByteBuffer();
        DataType outputDataType = DataType.getLarger(aDataType, bDataType);

        validateResult(outputDataType, cBuffer, expectedResultBuffer, nElementsC);
    }

    private void validateResult(DataType outputDataType, ByteBuffer cBuffer, ByteBuffer expectedResultBuffer, long nElementsC) {
        switch (outputDataType) {
            case INT8 -> {
                byte[] cArray = new byte[(int) nElementsC];
                cBuffer.get(cArray);
                byte[] expectedResultArray = new byte[(int) nElementsC];
                expectedResultBuffer.get(expectedResultArray);

                assertArrayEquals(expectedResultArray, cArray);
            }
            case INT16 -> {
                short[] cArray = new short[(int) nElementsC];
                cBuffer.asShortBuffer().get(cArray);
                short[] expectedResultArray = new short[(int) nElementsC];
                expectedResultBuffer.asShortBuffer().get(expectedResultArray);

                assertArrayEquals(expectedResultArray, cArray);
            }
            case INT32 -> {
                int[] cArray = new int[(int) nElementsC];
                cBuffer.asIntBuffer().get(cArray);
                int[] expectedResultArray = new int[(int) nElementsC];
                expectedResultBuffer.asIntBuffer().get(expectedResultArray);

                assertArrayEquals(expectedResultArray, cArray);
            }
            case INT64 -> {
                long[] cArray = new long[(int) nElementsC];
                cBuffer.asLongBuffer().get(cArray);
                long[] expectedResultArray = new long[(int) nElementsC];
                expectedResultBuffer.asLongBuffer().get(expectedResultArray);

                assertArrayEquals(expectedResultArray, cArray);
            }
            case FLOAT32 -> {
                float[] cArray = new float[(int) nElementsC];
                cBuffer.asFloatBuffer().get(cArray);
                float[] expectedResultArray = new float[(int) nElementsC];
                expectedResultBuffer.asFloatBuffer().get(expectedResultArray);

                for (int i = 0; i < nElementsC; i++) {
                    assertEquals(expectedResultArray[i], cArray[i], Math.ulp(expectedResultArray[i]));
                }
            }
            case FLOAT64 -> {
                double[] cArray = new double[(int) nElementsC];
                cBuffer.asDoubleBuffer().get(cArray);
                double[] expectedResultArray = new double[(int) nElementsC];
                expectedResultBuffer.asDoubleBuffer().get(expectedResultArray);

                for (int i = 0; i < nElementsC; i++) {
                    assertEquals(expectedResultArray[i], cArray[i], Math.ulp(expectedResultArray[i]));
                }
            }
        }
    }
}