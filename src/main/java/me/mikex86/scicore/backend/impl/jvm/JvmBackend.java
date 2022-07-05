package me.mikex86.scicore.backend.impl.jvm;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.backend.ScalarImpl;
import me.mikex86.scicore.backend.SciCoreBackend;
import me.mikex86.scicore.backend.ITensorImpl;
import org.jetbrains.annotations.NotNull;

public class JvmBackend implements SciCoreBackend {

    @Override
    public @NotNull ITensorImpl createTensor(@NotNull DataType dataType, long @NotNull [] shape) {
        return new JvmTensorImpl(dataType, shape);
    }

    @Override
    public @NotNull ScalarImpl createScalar(byte value) {
        return new JvmScalarImpl(value);
    }

    @Override
    public @NotNull ScalarImpl createScalar(short value) {
        return new JvmScalarImpl(value);
    }

    @Override
    public @NotNull ScalarImpl createScalar(int value) {
        return new JvmScalarImpl(value);
    }

    @Override
    public @NotNull ScalarImpl createScalar(long value) {
        return new JvmScalarImpl(value);
    }

    @Override
    public @NotNull ScalarImpl createScalar(float value) {
        return new JvmScalarImpl(value);
    }

    @Override
    public @NotNull ScalarImpl createScalar(double value) {
        return new JvmScalarImpl(value);
    }

}
