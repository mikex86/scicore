package me.mikex86.scicore.backend.impl.cuda.kernel;

import me.mikex86.scicore.tensor.DataType;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.List;

public class KernelNameUtility {

    @NotNull
    public static String getTypePermutation(@NotNull String functionName, @NotNull DataType dataTypeA, @NotNull DataType dataTypeB) {
        return functionName + "_" + getShortName(dataTypeA) + getShortName(dataTypeB);
    }

    @NotNull
    private static String getShortName(@NotNull DataType dataType) {
        return switch (dataType) {
            case INT8 -> "i8";
            case INT16 -> "i16";
            case INT32 -> "i32";
            case INT64 -> "i64";
            case FLOAT32 -> "f32";
            case FLOAT64 -> "f64";
            case BOOLEAN -> "b";
        };
    }

    @NotNull
    public static List<String> getAllTypePermutations(@NotNull String functionName, @NotNull List<DataType> dataTypes) {
        List<String> permutations = new ArrayList<>(dataTypes.size() * dataTypes.size());
        for (DataType a : dataTypes) {
            for (DataType b : dataTypes) {
                permutations.add(getTypePermutation(functionName, a, b));
            }
        }
        return permutations;
    }
}
