package me.mikex86.scicore;

import org.jetbrains.annotations.NotNull;

public enum DataType {

    INT8(8, false, true), INT16(16, false, true), INT32(32, false, true), INT64(64, false, true),
    FLOAT32(32, true, true), FLOAT64(64, true, true),
    BOOLEAN(1, false, true);

    private final int bits;
    private final boolean isFp;

    private final boolean isNumeric;

    DataType(int bits, boolean isFp, boolean isNumeric) {
        this.bits = bits;
        this.isFp = isFp;
        this.isNumeric = isNumeric;

        if (isFp && !isNumeric) {
            throw new IllegalArgumentException("Floating point data types must be numeric");
        }
    }

    public boolean isFloatingPoint() {
        return isFp;
    }

    public boolean isNumeric() {
        return isNumeric;
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

    public boolean isSameType(@NotNull Class<?> fClass) {
        if (fClass == byte.class || fClass == Byte.class) {
            return this == INT8;
        } else if (fClass == short.class || fClass == Short.class) {
            return this == INT16;
        } else if (fClass == int.class || fClass == Integer.class) {
            return this == INT32;
        } else if (fClass == long.class || fClass == Long.class) {
            return this == INT64;
        } else if (fClass == float.class || fClass == Float.class) {
            return this == FLOAT32;
        } else if (fClass == double.class || fClass == Double.class) {
            return this == FLOAT64;
        } else if (fClass == boolean.class || fClass == Boolean.class) {
            return this == BOOLEAN;
        } else {
            return false;
        }
    }

    public int getBits() {
        return bits;
    }

    public int getSize(){
        return bits / 8;
    }
}