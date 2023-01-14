package me.mikex86.scicore.backend.impl.cuda.codegen;

import me.mikex86.scicore.tensor.DataType;
import org.jetbrains.annotations.NotNull;

public class DataTypeUtils {

    @NotNull
    public static String getCudaType(@NotNull DataType type) {
        return switch (type) {
            case INT8 -> "int8_t";
            case INT16 -> "int16_t";
            case INT32 -> "int32_t";
            case INT64 -> "int64_t";
            case FLOAT32 -> "float";
            case FLOAT64 -> "double";
            default -> throw new IllegalArgumentException("Unsupported type for cuda code generator: " + type);
        };
    }

}
