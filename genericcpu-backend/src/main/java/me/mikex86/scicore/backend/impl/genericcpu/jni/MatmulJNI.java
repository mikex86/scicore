package me.mikex86.scicore.backend.impl.genericcpu.jni;

public class MatmulJNI {

    public static final int OP_NONE = 0;
    public static final int OP_TRANSPOSE = 1;

    public static final int DATA_TYPE_INT8 = 1;
    public static final int DATA_TYPE_INT16 = 2;
    public static final int DATA_TYPE_INT32 = 3;
    public static final int DATA_TYPE_INT64 = 4;
    public static final int DATA_TYPE_FLOAT32 = 5;
    public static final int DATA_TYPE_FLOAT64 = 6;

    public static native void matmul(int transa, int transb,
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

}
