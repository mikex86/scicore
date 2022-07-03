package me.mikex86.scicore;

import me.mikex86.scicore.backend.ScalarImpl;
import me.mikex86.scicore.backend.SciCoreBackend;
import org.jetbrains.annotations.NotNull;

public class Scalar implements IScalar {

    @NotNull
    private final ScalarImpl scalarImpl;

    Scalar(@NotNull SciCoreBackend backend, byte value) {
        this.scalarImpl = backend.createScalar(value);
    }

    Scalar(@NotNull SciCoreBackend backend, short value) {
        this.scalarImpl = backend.createScalar(value);
    }

    Scalar(@NotNull SciCoreBackend backend, int value) {
        this.scalarImpl = backend.createScalar(value);
    }

    Scalar(@NotNull SciCoreBackend backend, long value) {
        this.scalarImpl = backend.createScalar(value);
    }

    Scalar(@NotNull SciCoreBackend backend, float value) {
        this.scalarImpl = backend.createScalar(value);
    }

    Scalar(@NotNull SciCoreBackend backend, double value) {
        this.scalarImpl = backend.createScalar(value);
    }

    @NotNull
    public ScalarImpl getScalarImpl() {
        return scalarImpl;
    }

    @Override
    @NotNull
    public DataType getDataType() {
        return scalarImpl.getDataType();
    }
}
