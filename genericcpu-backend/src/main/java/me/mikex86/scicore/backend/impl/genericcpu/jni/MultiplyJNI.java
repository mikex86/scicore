package me.mikex86.scicore.backend.impl.genericcpu.jni;

import me.mikex86.scicore.DataType;
import org.jetbrains.annotations.NotNull;

public class MultiplyJNI {

    private static final int DATA_TYPE_INT8 = 1;
    private static final int DATA_TYPE_INT16 = 2;
    private static final int DATA_TYPE_INT32 = 3;
    private static final int DATA_TYPE_INT64 = 4;
    private static final int DATA_TYPE_FLOAT32 = 5;
    private static final int DATA_TYPE_FLOAT64 = 6;

    private static native void nmultiply(long aPtr, long[] shapeA, long[] stridesA, int dataTypeA,
                                         long bPtr, long[] shapeB, long[] stridesB, int dataTypeB,
                                         long cPtr, long[] shapeC, long[] stridesC, int dataTypeC);

    public static void multiply(long aPtr, long[] shapeA, long[] stridesA, DataType dataTypeA,
                                long bPtr, long[] shapeB, long[] stridesB, DataType dataTypeB,
                                long cPtr, long[] shapeC, long[] stridesC, DataType dataTypeC) {
        nmultiply(
                aPtr, shapeA, stridesA, dataTypeToInt(dataTypeA),
                bPtr, shapeB, stridesB, dataTypeToInt(dataTypeB),
                cPtr, shapeC, stridesC, dataTypeToInt(dataTypeC)
        );
    }

    private static int dataTypeToInt(@NotNull DataType dataTypeA) {
        return switch (dataTypeA) {
            case INT8 -> DATA_TYPE_INT8;
            case INT16 -> DATA_TYPE_INT16;
            case INT32 -> DATA_TYPE_INT32;
            case INT64 -> DATA_TYPE_INT64;
            case FLOAT32 -> DATA_TYPE_FLOAT32;
            case FLOAT64 -> DATA_TYPE_FLOAT64;
            case BOOLEAN -> throw new IllegalArgumentException("Boolean data type is not supported");
        };
    }

}
