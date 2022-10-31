package me.mikex86.scicore.backend.impl.genericcpu.jni;

import me.mikex86.scicore.tensor.DataType;
import org.jetbrains.annotations.NotNull;

public class JNIDataTypes {

    public static final int DATA_TYPE_INT8 = 1;
    public static final int DATA_TYPE_INT16 = 2;
    public static final int DATA_TYPE_INT32 = 3;
    public static final int DATA_TYPE_INT64 = 4;
    public static final int DATA_TYPE_FLOAT32 = 5;
    public static final int DATA_TYPE_FLOAT64 = 6;

    public static int dataTypeToInt(@NotNull DataType dataType) {
        return switch (dataType) {
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
