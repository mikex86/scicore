package me.mikex86.scicore;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.lang.reflect.InvocationTargetException;
import java.util.List;

public interface ISciCore {
    static boolean isBackendAvailable(@NotNull BackendType backendType) {
        try {
            backendType.newInstance();
            return true;
        } catch (RuntimeException t) {
            return false;
        }
    }

    @NotNull ITensor zeros(@NotNull DataType dataType, long @NotNull ... shape);

    void seed(long seed);

    @NotNull ITensor uniform(@NotNull DataType dataType, long @NotNull ... shape);

    @NotNull ITensor randint(@NotNull DataType dataType, long min, long max, long @NotNull ... shape);

    @NotNull ITensor gaussian(@NotNull DataType dataType, long @NotNull ... shape);

    void addBackend(@NotNull BackendType backendType);

    @NotNull ISciCoreBackend getBackend();

    void fill(@NotNull ITensor tensor, byte value);

    void fill(@NotNull ITensor tensor, short value);

    void fill(@NotNull ITensor tensor, int value);

    void fill(@NotNull ITensor tensor, long value);

    void fill(@NotNull ITensor tensor, float value);

    void fill(@NotNull ITensor tensor, double value);

    void fill(@NotNull ITensor tensor, boolean value);

    @NotNull ITensor matmul(@NotNull ITensor a, @NotNull ITensor b);

    @NotNull ITensor matmul(@NotNull ITensor a, @NotNull ITensor b, boolean transposeA, boolean transposeB);

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

    @NotNull ITensor zerosLike(@NotNull ITensor reference);

    @NotNull ITensor onesLike(@NotNull ITensor reference);

    @NotNull IGraph getExecutionGraphUpTo(@NotNull ITensor root);

    @NotNull IGraph getBackpropagationGraphUpTo(@NotNull ITensor root, @NotNull List<ITensor> parameters);

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
     *
     * @param start    the start of the interval.
     * @param stop     the end of the interval.
     * @param step     the spacing between values.
     * @param dataType the data type of the resulting tensor.
     * @return the tensor with evenly spaced values.
     */
    @NotNull ITensor arange(long start, long stop, long step, @NotNull DataType dataType);

    @NotNull ITensor arange(double start, double stop, double step, @NotNull DataType dataType);

    /**
     * Calculates cross-entropy loss between the given logits and labels.
     *
     * @param logits      the logits tensor (unnormalized log probabilities).
     * @param target      the target tensor (integers).
     * @param ignoreIndex the index to ignore in the target tensor. (must be negative)
     * @return the cross-entropy loss.
     */
    @NotNull ITensor crossEntropy(@NotNull ITensor logits, @NotNull ITensor target, @Nullable Long ignoreIndex);

    @NotNull
    default ITensor crossEntropy(@NotNull ITensor input, @NotNull ITensor target) {
        return crossEntropy(input, target, null);
    }

    @NotNull
    default ITensor stack(int dimension, @NotNull List<ITensor> tensors) {
        return stack(dimension, tensors.toArray(new ITensor[0]));
    }

    @NotNull
    ITensor stack(int dimension, @NotNull ITensor @NotNull ... tensors);

    @NotNull
    ITensor multinomial(@NotNull ITensor input, long numSamples);

    /**
     * Disables fallback to lower priority backends, when an implementation for a given operation is not available
     * in the current backend.
     */
    void disableBackendFallback();

    @NotNull ITensor triangle(@NotNull DataType dataType, long dim0, long dim1, double topValue, double bottomValue);

    enum BackendType {

        JVM {
            @NotNull
            @Override
            public ISciCoreBackend newInstance() {
                return new JvmBackend();
            }
        },

        CPU {
            @NotNull
            @Override
            public ISciCoreBackend newInstance() {
                Class<?> cudaBackendClass;
                try {
                    cudaBackendClass = Class.forName("me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend");
                } catch (ClassNotFoundException e) {
                    throw new RuntimeException("Generic-CPU backend not found. Make sure you have the Generic-CPU backend dependency in your classpath.");
                }
                try {
                    return (ISciCoreBackend) cudaBackendClass.getConstructor().newInstance();
                } catch (InstantiationException | IllegalAccessException |
                         NoSuchMethodException | InvocationTargetException e) {
                    throw new RuntimeException("Could not instantiate generic-cpu backend.", e);
                }
            }
        },

        CUDA {
            @Override
            public @NotNull ISciCoreBackend newInstance() {
                Class<?> cudaBackendClass;
                try {
                    cudaBackendClass = Class.forName("me.mikex86.scicore.backend.impl.cuda.CudaBackend");
                } catch (ClassNotFoundException e) {
                    throw new RuntimeException("CUDA backend not found. Make sure you have the CUDA backend dependency in your classpath.");
                }
                try {
                    return (ISciCoreBackend) cudaBackendClass.getConstructor().newInstance();
                } catch (InstantiationException | IllegalAccessException |
                         NoSuchMethodException | InvocationTargetException e) {
                    throw new RuntimeException("Could not instantiate CUDA backend.", e);
                }
            }
        };

        @NotNull
        public abstract ISciCoreBackend newInstance();
    }

    boolean isTraining();

    void eval();

    void train();
}
