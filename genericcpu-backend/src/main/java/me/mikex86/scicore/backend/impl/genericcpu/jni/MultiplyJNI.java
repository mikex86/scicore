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
            DataType aDataType,
            long nElementsA,
            long bPtr,
            DataType bDataType,
            long nElementsB,
            long cPtr,
            long nElementsC,
            DataType cDataType
    ) {
        int aDataTypeInt = getDataTypeInt(aDataType).orElseThrow(() -> new IllegalArgumentException("Unsupported data type: " + aDataType));
        int bDataTypeInt = getDataTypeInt(bDataType).orElseThrow(() -> new IllegalArgumentException("Invalid data type: " + nElementsB));
        DataType resultType = DataType.getLarger(aDataType, bDataType);
        Validator.assertTrue(cDataType == resultType, "Multiplying  " + aDataType + " and " + bDataType + " results in " + resultType + " but the result tensor is of type " + cDataType);
        nmultiply(aPtr, aDataTypeInt, nElementsA, bPtr, bDataTypeInt, nElementsB, cPtr, nElementsC);
    }

    @NotNull
    public static OptionalInt getDataTypeInt(@NotNull DataType dataType) {
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
