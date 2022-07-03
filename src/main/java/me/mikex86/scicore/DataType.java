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