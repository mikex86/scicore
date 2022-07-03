package me.mikex86.scicore.backend;

import me.mikex86.scicore.DataType;
import org.jetbrains.annotations.NotNull;

public interface ScalarImpl {

    @NotNull DataType getDataType();

    byte getByte();

    short getShort();

    int getInt();

    long getLong();

    float getFloat();

    double getDouble();
}
