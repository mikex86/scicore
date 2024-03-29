package me.mikex86.scicore.backend.impl.genericcpu.jni;

import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;
import org.lwjgl.system.MemoryUtil;

import static java.lang.Math.max;

public class MatmulJNI {

    public static final int MATMUL_LAYOUT_ROW_MAJOR = 0;
    public static final int MATMUL_LAYOUT_COLUMN_MAJOR = 1;

    public static final int MATMUL_OP_NONE = 0;
    public static final int MATMUL_OP_TRANSPOSE = 1;

    public static final int MATMUL_DATA_TYPE_INT8 = 1;
    public static final int MATMUL_DATA_TYPE_INT16 = 2;
    public static final int MATMUL_DATA_TYPE_INT32 = 3;
    public static final int MATMUL_DATA_TYPE_INT64 = 4;
    public static final int MATMUL_DATA_TYPE_FLOAT32 = 5;
    public static final int MATMUL_DATA_TYPE_FLOAT64 = 6;

    /**
     * A pointer pointing to a byte value of 1.
     */
    public static final long INT8_ALPHA_IDENTITY;

    /**
     * A pointer pointing to a short value of 1.
     */
    public static final long INT16_ALPHA_IDENTITY;

    /**
     * A pointer pointing to an int value of 1.
     */
    public static final long INT32_ALPHA_IDENTITY;

    /**
     * A pointer pointing to a long value of 1.
     */
    public static final long INT64_ALPHA_IDENTITY;

    /**
     * A pointer pointing to a float value of 1.0.
     */
    public static final long FLOAT_ALPHA_IDENTITY;

    /**
     * A pointer pointing to a double value of 1.0.
     */
    public static final long DOUBLE_ALPHA_IDENTITY;

    static {
        INT8_ALPHA_IDENTITY = MemoryUtil.nmemAlloc(1);
        MemoryUtil.memPutByte(INT8_ALPHA_IDENTITY, (byte) 1);

        INT16_ALPHA_IDENTITY = MemoryUtil.nmemAlloc(2);
        MemoryUtil.memPutShort(INT16_ALPHA_IDENTITY, (short) 1);

        INT32_ALPHA_IDENTITY = MemoryUtil.nmemAlloc(4);
        MemoryUtil.memPutInt(INT32_ALPHA_IDENTITY, 1);

        INT64_ALPHA_IDENTITY = MemoryUtil.nmemAlloc(8);
        MemoryUtil.memPutLong(INT64_ALPHA_IDENTITY, 1L);

        FLOAT_ALPHA_IDENTITY = MemoryUtil.nmemAlloc(4);
        MemoryUtil.memPutFloat(FLOAT_ALPHA_IDENTITY, 1.0f);

        DOUBLE_ALPHA_IDENTITY = MemoryUtil.nmemAlloc(8);
        MemoryUtil.memPutDouble(DOUBLE_ALPHA_IDENTITY, 1.0);
    }

    private static native void nmatmul(
            int layout,
            int transa, int transb,
            int m, int n, int k,
            long alphaPtr,
            long aPtr,
            int aType,
            int lda,
            long betaPtr, long bPtr,
            int bType,
            int ldb,
            long cPtr,
            int cType,
            int ldc);

    public static void matmul(
            int layout,
            int transa, int transb,
            int m, int n, int k,
            long alphaPtr,
            long aPtr,
            int aType,
            int lda,
            long betaPtr, long bPtr,
            int bType,
            int ldb,
            long cPtr,
            int cType,
            int ldc) {
        Validator.assertTrue(transa == MATMUL_OP_NONE || transa == MATMUL_OP_TRANSPOSE, "transa must be OP_NONE or OP_TRANSPOSE");
        Validator.assertTrue(transb == MATMUL_OP_NONE || transb == MATMUL_OP_TRANSPOSE, "transb must be OP_NONE or OP_TRANSPOSE");
        Validator.assertTrue(aType == MATMUL_DATA_TYPE_INT8 || aType == MATMUL_DATA_TYPE_INT16 || aType == MATMUL_DATA_TYPE_INT32 || aType == MATMUL_DATA_TYPE_INT64 || aType == MATMUL_DATA_TYPE_FLOAT32 || aType == MATMUL_DATA_TYPE_FLOAT64, "aType must be DATA_TYPE_INT8, DATA_TYPE_INT16, DATA_TYPE_INT32, DATA_TYPE_INT64, DATA_TYPE_FLOAT32 or DATA_TYPE_FLOAT64");
        Validator.assertTrue(bType == MATMUL_DATA_TYPE_INT8 || bType == MATMUL_DATA_TYPE_INT16 || bType == MATMUL_DATA_TYPE_INT32 || bType == MATMUL_DATA_TYPE_INT64 || bType == MATMUL_DATA_TYPE_FLOAT32 || bType == MATMUL_DATA_TYPE_FLOAT64, "bType must be DATA_TYPE_INT8, DATA_TYPE_INT16, DATA_TYPE_INT32, DATA_TYPE_INT64, DATA_TYPE_FLOAT32 or DATA_TYPE_FLOAT64");
        Validator.assertTrue(cType == MATMUL_DATA_TYPE_INT8 || cType == MATMUL_DATA_TYPE_INT16 || cType == MATMUL_DATA_TYPE_INT32 || cType == MATMUL_DATA_TYPE_INT64 || cType == MATMUL_DATA_TYPE_FLOAT32 || cType == MATMUL_DATA_TYPE_FLOAT64, "cType must be DATA_TYPE_INT8, DATA_TYPE_INT16, DATA_TYPE_INT32, DATA_TYPE_INT64, DATA_TYPE_FLOAT32 or DATA_TYPE_FLOAT64");
        Validator.assertTrue(m >= 0, "m must be greater than or equal to 0");
        Validator.assertTrue(n >= 0, "n must be greater than or equal to 0");
        Validator.assertTrue(k >= 0, "k must be greater than or equal to 0");
        Validator.assertTrue(lda > 0, "ldc must be greater than 0");
        Validator.assertTrue(ldb > 0, "ldc must be greater than 0");
        Validator.assertTrue(ldc > 0, "ldc must be greater than 0");

        nmatmul(layout, transa, transb, m, n, k, alphaPtr, aPtr, aType, lda, betaPtr, bPtr, bType, ldb, cPtr, cType, ldc);
    }

    public static void matmul(
            int layout,
            int transa, int transb,
            int m, int n, int k,
            long aPtr,
            int aType,
            int lda,
            long bPtr,
            int bType,
            int ldb,
            long cPtr,
            int cType,
            int ldc) {
        long alphaIdentity = getIdentity(aType), betaIdentity = getIdentity(bType);
        matmul(layout, transa, transb, m, n, k, alphaIdentity, aPtr, aType, lda, betaIdentity, bPtr, bType, ldb, cPtr, cType, ldc);
    }

    private static long getIdentity(int type) {
        return switch (type) {
            case MATMUL_DATA_TYPE_INT8 -> INT8_ALPHA_IDENTITY;
            case MATMUL_DATA_TYPE_INT16 -> INT16_ALPHA_IDENTITY;
            case MATMUL_DATA_TYPE_INT32 -> INT32_ALPHA_IDENTITY;
            case MATMUL_DATA_TYPE_INT64 -> INT64_ALPHA_IDENTITY;
            case MATMUL_DATA_TYPE_FLOAT32 -> FLOAT_ALPHA_IDENTITY;
            case MATMUL_DATA_TYPE_FLOAT64 -> DOUBLE_ALPHA_IDENTITY;
            default -> throw new IllegalArgumentException("Unknown data type: " + type);
        };
    }

    public static int getMatmulDataType(@NotNull DataType dataType) {
        return switch (dataType) {
            case INT8 -> MATMUL_DATA_TYPE_INT8;
            case INT16 -> MATMUL_DATA_TYPE_INT16;
            case INT32 -> MATMUL_DATA_TYPE_INT32;
            case INT64 -> MATMUL_DATA_TYPE_INT64;
            case FLOAT32 -> MATMUL_DATA_TYPE_FLOAT32;
            case FLOAT64 -> MATMUL_DATA_TYPE_FLOAT64;
            default -> throw new IllegalStateException("Unsupported data type: " + dataType);
        };
    }
}
