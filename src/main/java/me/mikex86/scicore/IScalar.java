package me.mikex86.scicore;

import org.jetbrains.annotations.NotNull;

public interface IScalar extends IValue {
    @NotNull DataType getDataType();
}
