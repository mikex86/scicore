package me.mikex86.scicore;

import org.jetbrains.annotations.NotNull;

public enum DataType {

    INT8(8, false), INT16(16, false), INT32(32, false), INT64(64, false), FLOAT32(32, true), FLOAT64(64, true);

    private final int bits;
    private final boolean isFp;

    DataType(int bits, boolean isFp) {
        this.bits = bits;
        this.isFp = isFp;
    }

    public boolean isFloatingPoint() {
        return isFp;
    }

    @NotNull
    public static DataType fromClass(@NotNull Class<?> componentClass) {
        if (componentClass == byte.class) {
            return INT8;
        } else if (componentClass == short.class) {
            return INT16;
        } else if (componentClass == int.class) {
            return INT32;
        } else if (componentClass == long.class) {
            return INT64;
        } else if (componentClass == float.class) {
            return FLOAT32;
        } else if (componentClass == double.class) {
            return FLOAT64;
        } else {
            throw new IllegalArgumentException("Unsupported component class: " + componentClass);
        }
    }

    @NotNull
    public static DataType getLarger(DataType a, DataType b) {
        DataType largerByBitSize = a.bits > b.bits ? a : b;
        if (a.isFp) {
            if (b.isFp) {
                return largerByBitSize;
            } else {
                return a;
            }
        }
        return largerByBitSize;
    }
}