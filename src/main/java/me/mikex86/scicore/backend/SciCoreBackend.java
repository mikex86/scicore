package me.mikex86.scicore.backend;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import org.jetbrains.annotations.NotNull;

public interface SciCoreBackend {

    @NotNull
    TensorImpl createTensor(@NotNull DataType dataType, long @NotNull [] shape);

    @NotNull
    ScalarImpl createScalar(byte value);

    @NotNull
    ScalarImpl createScalar(short value);

    @NotNull
    ScalarImpl createScalar(int value);

    @NotNull
    ScalarImpl createScalar(long value);

    @NotNull
    ScalarImpl createScalar(float value);

    @NotNull
    ScalarImpl createScalar(double value);
}
