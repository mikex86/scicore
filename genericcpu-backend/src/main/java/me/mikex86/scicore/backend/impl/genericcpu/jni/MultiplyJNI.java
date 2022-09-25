package me.mikex86.scicore.backend.impl.genericcpu.jni;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

import java.util.OptionalInt;

public class MultiplyJNI {

    public static final int MULTIPLY_DATA_TYPE_INT8 = 1;
    public static final int MULTIPLY_DATA_TYPE_INT16 = 2;
    public static final int MULTIPLY_DATA_TYPE_INT32 = 3;
    public static final int MULTIPLY_DATA_TYPE_INT64 = 4;
    public static final int MULTIPLY_DATA_TYPE_FLOAT32 = 5;
    public static final int MULTIPLY_DATA_TYPE_FLOAT64 = 6;

    private static native void nmultiply(
            long aPtr,
            int aDataType,
            long nElementsA,
            long bPtr,
            int bDataType,
            long nElementsB,
            long cPtr,
            long nElementsC
    );

    public static void multiply(
            long aPtr,
            int aDataType,
            long nElementsA,
            long bPtr,
            int bDataType,
            long nElementsB,
            long cPtr,
            long nElementsC
    ) {
        Validator.assertTrue(aDataType == MULTIPLY_DATA_TYPE_INT8 || aDataType == MULTIPLY_DATA_TYPE_INT16 || aDataType == MULTIPLY_DATA_TYPE_INT32 || aDataType == MULTIPLY_DATA_TYPE_INT64 || aDataType == MULTIPLY_DATA_TYPE_FLOAT32 || aDataType == MULTIPLY_DATA_TYPE_FLOAT64, "Invalid data type");
        Validator.assertTrue(bDataType == MULTIPLY_DATA_TYPE_INT8 || bDataType == MULTIPLY_DATA_TYPE_INT16 || bDataType == MULTIPLY_DATA_TYPE_INT32 || bDataType == MULTIPLY_DATA_TYPE_INT64 || bDataType == MULTIPLY_DATA_TYPE_FLOAT32 || bDataType == MULTIPLY_DATA_TYPE_FLOAT64, "Invalid data type");
        nmultiply(aPtr, aDataType, nElementsA, bPtr, bDataType, nElementsB, cPtr, nElementsC);
    }

    @NotNull
    public static OptionalInt getDataType(@NotNull DataType dataType) {
        return switch (dataType) {
            case INT8 -> OptionalInt.of(MULTIPLY_DATA_TYPE_INT8);
            case INT16 -> OptionalInt.of(MULTIPLY_DATA_TYPE_INT16);
            case INT32 -> OptionalInt.of(MULTIPLY_DATA_TYPE_INT32);
            case INT64 -> OptionalInt.of(MULTIPLY_DATA_TYPE_INT64);
            case FLOAT32 -> OptionalInt.of(MULTIPLY_DATA_TYPE_FLOAT32);
            case FLOAT64 -> OptionalInt.of(MULTIPLY_DATA_TYPE_FLOAT64);
            default -> OptionalInt.empty();
        };
    }
}
