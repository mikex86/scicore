package me.mikex86.scicore;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.op.IGraph;
import org.jetbrains.annotations.NotNull;

public interface ISciCore {
    @NotNull ITensor zeros(@NotNull DataType dataType, long @NotNull ... shape);

    void seed(long seed);

    @NotNull ITensor uniform(@NotNull DataType dataType, long @NotNull ... shape);

    void setBackend(@NotNull BackendType backendType);

    @NotNull ISciCoreBackend getBackend();

    void fill(@NotNull ITensor tensor, byte value);

    void fill(@NotNull ITensor tensor, short value);

    void fill(@NotNull ITensor tensor, int value);

    void fill(@NotNull ITensor tensor, long value);

    void fill(@NotNull ITensor tensor, float value);

    void fill(@NotNull ITensor tensor, double value);

    void fill(@NotNull ITensor tensor, boolean value);

    @NotNull ITensor matmul(@NotNull ITensor a, @NotNull ITensor b);

    @NotNull ITensor exp(@NotNull ITensor a);

    @NotNull ITensor softmax(@NotNull ITensor input, int dimension);

    @NotNull ITensor reduceSum(@NotNull ITensor input, int dimension);

    @NotNull ITensor array(byte @NotNull [] array);

    @NotNull ITensor array(short @NotNull [] array);

    @NotNull ITensor array(int @NotNull [] array);

    @NotNull ITensor array(long @NotNull [] array);

    @NotNull ITensor array(float @NotNull [] array);

    @NotNull ITensor array(double @NotNull [] array);

    @NotNull ITensor array(boolean @NotNull [] array);

    @NotNull ITensor matrix(byte @NotNull [][] array);

    @NotNull ITensor matrix(short @NotNull [][] array);

    @NotNull ITensor matrix(int @NotNull [][] array);

    @NotNull ITensor matrix(long @NotNull [][] array);

    @NotNull ITensor matrix(float @NotNull [][] array);

    @NotNull ITensor matrix(double @NotNull [][] array);

    @NotNull ITensor matrix(boolean @NotNull [][] array);

    @NotNull ITensor ndarray(Object array);

    @NotNull ITensor onesLike(@NotNull ITensor reference);

    @NotNull IGraph getGraphUpTo(@NotNull ITensor tensor);

    @NotNull ITensor scalar(byte value);

    @NotNull ITensor scalar(short value);

    @NotNull ITensor scalar(int value);

    @NotNull ITensor scalar(long value);

    @NotNull ITensor scalar(float value);

    @NotNull ITensor scalar(double value);

    @NotNull ITensor scalar(boolean value);

    @NotNull ITensor pow(ITensor base, byte exponent);

    @NotNull ITensor pow(ITensor base, short exponent);

    @NotNull ITensor pow(ITensor base, int exponent);

    @NotNull ITensor pow(ITensor base, long exponent);

    @NotNull ITensor pow(ITensor base, float exponent);

    @NotNull ITensor pow(ITensor base, double exponent);

    /**
     * Returns evenly spaced values within a given interval.
     * @param start the start of the interval.
     * @param stop the end of the interval.
     * @param step the step size.
     * @param shape the shape of the resulting tensor.
     * @param dataType the data type of the resulting tensor.
     * @return the tensor with evenly spaced values.
     */
    @NotNull ITensor arange(double start, double stop, double step, long @NotNull [] shape, @NotNull DataType dataType);

    enum BackendType {

        JVM {
            @NotNull
            @Override
            public ISciCoreBackend newInstance() {
                return new JvmBackend();
            }
        };

        @NotNull
        public abstract ISciCoreBackend newInstance();
    }
}
