package me.mikex86.scicore.tensor;

import org.jetbrains.annotations.NotNull;

import java.util.Optional;

public enum DataType {

    INT8(8, false, true, true), INT16(16, false, true, true),
    INT32(32, false, true, true), INT64(64, false, true, true),
    FLOAT32(32, true, false, true), FLOAT64(64, true, false, true),
    BOOLEAN(1, false, false, false);

    private final int bits;
    private final boolean isFloatingPoint;

    private final boolean isNumeric;
    private final boolean integer;

    DataType(int bits, boolean isFloatingPoint, boolean integer, boolean isNumeric) {
        this.bits = bits;
        this.isFloatingPoint = isFloatingPoint;
        this.integer = integer;
        this.isNumeric = isNumeric;

        if (isFloatingPoint && !isNumeric) {
            throw new IllegalArgumentException("Floating point data types must be numeric");
        }
    }

    @NotNull
    public static Optional<DataType> fromOrdinal(int ordinal) {
        if (ordinal < 0 || ordinal >= values().length) {
            return Optional.empty();
        }
        return Optional.of(values()[ordinal]);
    }

    public boolean isFloatingPoint() {
        return isFloatingPoint;
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
        } else if (componentClass == boolean.class) {
            return BOOLEAN;
        } else {
            throw new IllegalArgumentException("Unsupported component class: " + componentClass);
        }
    }

    @NotNull
    public static Optional<DataType> findByName(@NotNull String name) {
        for (DataType dataType : values()) {
            if (dataType.name().equalsIgnoreCase(name)) {
                return Optional.of(dataType);
            }
        }
        return Optional.empty();
    }

    @NotNull
    public static DataType getLarger(DataType a, DataType b) {
        DataType largerByBitSize = a.bits > b.bits ? a : b;
        if (a.isFloatingPoint && b.isFloatingPoint) {
            return largerByBitSize;
        } else if (a.isFloatingPoint) {
            return a;
        } else if (b.isFloatingPoint) {
            return b;
        } else {
            return largerByBitSize;
        }
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

    public int getSize() {
        return bits / 8;
    }

    public long getSizeOf(long nElements) {
        return (nElements * bits + 7) / 8;
    }

    public boolean isInteger() {
        return integer;
    }
}