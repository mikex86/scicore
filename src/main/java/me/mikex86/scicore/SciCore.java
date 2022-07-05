package me.mikex86.scicore;

import me.mikex86.scicore.backend.SciCoreBackend;
import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.utils.ArrayUtils;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.Objects;
import java.util.Random;

public class SciCore {

    @Nullable
    private SciCoreBackend sciCoreBackend = null;

    @NotNull
    public ITensor zeros(@NotNull DataType dataType, long @NotNull ... shape) {
        return new Tensor(dataType, shape, getBackend());
    }

    @NotNull
    public ITensor uniform(@NotNull DataType dataType, long @NotNull ... shape) {
        Tensor tensor = new Tensor(dataType, shape, getBackend());
        long numberOfElements = tensor.getNumberOfElements();
        Random random = new Random();
        switch (dataType) {
            case INT8 -> {
                for (long i = 0; i < numberOfElements; i++) {
                    tensor.setByteFlat((byte) (random.nextInt(255) - 128), i);
                }
            }
            case INT16 -> {
                for (long i = 0; i < numberOfElements; i++) {
                    tensor.setShortFlat((short) (random.nextInt(65535) - 32768), i);
                }
            }
            case INT32 -> {
                for (long i = 0; i < numberOfElements; i++) {
                    tensor.setIntFlat(random.nextInt(), i);
                }
            }
            case INT64 -> {
                for (long i = 0; i < numberOfElements; i++) {
                    tensor.setLongFlat(random.nextLong(), i);
                }
            }
            case FLOAT32 -> {
                for (long i = 0; i < numberOfElements; i++) {
                    tensor.setFloatFlat(random.nextFloat(), i);
                }
            }
            case FLOAT64 -> {
                for (long i = 0; i < numberOfElements; i++) {
                    tensor.setDoubleFlat(random.nextDouble(), i);
                }
            }
            default -> throw new IllegalArgumentException("Unsupported data type: " + dataType);
        }
        return tensor;
    }

    @NotNull
    public ITensor multiplied(@NotNull ITensor a, @NotNull Scalar b) {
        return a.multiplied(b);
    }

    public void setBackend(@NotNull BackendType backendType) {
        if (sciCoreBackend != null) {
            throw new IllegalStateException("SciCore backend already initialized!");
        }
        sciCoreBackend = backendType.newInstance();
    }

    @NotNull
    public SciCoreBackend getBackend() {
        return Objects.requireNonNull(sciCoreBackend, "Backend has not yet been configured");
    }

    @NotNull
    public IScalar scalar(byte i) {
        return new Scalar(getBackend(), i);
    }

    @NotNull
    public IScalar scalar(short i) {
        return new Scalar(getBackend(), i);
    }

    @NotNull
    public IScalar scalar(int i) {
        return new Scalar(getBackend(), i);
    }

    @NotNull
    public IScalar scalar(float f) {
        return new Scalar(getBackend(), f);
    }

    @NotNull
    public IScalar scalar(double d) {
        return new Scalar(getBackend(), d);
    }

    public void fill(@NotNull ITensor tensor, byte i) {
        tensor.fill(i);
    }

    public void fill(@NotNull ITensor tensor, short i) {
        tensor.fill(i);
    }

    public void fill(@NotNull ITensor tensor, int i) {
        tensor.fill(i);
    }

    public void fill(@NotNull ITensor tensor, float f) {
        tensor.fill(f);
    }

    public void fill(@NotNull ITensor tensor, double d) {
        tensor.fill(d);
    }

    @NotNull
    public ITensor matmul(@NotNull ITensor a, @NotNull ITensor b) {
        return a.matmul(b);
    }

    @NotNull
    public ITensor exp(@NotNull ITensor a) {
        return a.exp();
    }

    @NotNull
    public ITensor softmax(@NotNull ITensor input, int dimension) {
        return input.softmax(dimension);
    }

    @NotNull
    public ITensor reduceSum(@NotNull ITensor input, int dimension) {
        return input.reduceSum(dimension);
    }

    @NotNull
    public ITensor array(byte @NotNull [] array) {
        ITensor tensor = new Tensor(DataType.INT8, new long[]{array.length}, getBackend());
        for (int i = 0; i < array.length; i++) {
            tensor.setByteFlat(array[i], i);
        }
        return tensor;
    }

    @NotNull
    public ITensor array(short @NotNull [] array) {
        ITensor tensor = new Tensor(DataType.INT16, new long[]{array.length}, getBackend());
        for (int i = 0; i < array.length; i++) {
            tensor.setShortFlat(array[i], i);
        }
        return tensor;
    }

    @NotNull
    public ITensor array(int @NotNull [] array) {
        ITensor tensor = new Tensor(DataType.INT32, new long[]{array.length}, getBackend());
        for (int i = 0; i < array.length; i++) {
            tensor.setIntFlat(array[i], i);
        }
        return tensor;
    }

    @NotNull
    public ITensor array(long @NotNull [] array) {
        ITensor tensor = new Tensor(DataType.INT64, new long[]{array.length}, getBackend());
        for (int i = 0; i < array.length; i++) {
            tensor.setLongFlat(array[i], i);
        }
        return tensor;
    }

    @NotNull
    public ITensor array(float @NotNull [] array) {
        ITensor tensor = new Tensor(DataType.FLOAT32, new long[]{array.length}, getBackend());
        for (int i = 0; i < array.length; i++) {
            tensor.setFloatFlat(array[i], i);
        }
        return tensor;
    }

    @NotNull
    public ITensor array(double @NotNull [] array) {
        ITensor tensor = new Tensor(DataType.FLOAT64, new long[]{array.length}, getBackend());
        for (int i = 0; i < array.length; i++) {
            tensor.setDoubleFlat(array[i], i);
        }
        return tensor;
    }

    @NotNull
    public ITensor matrix(byte @NotNull [][] array) {
        for (byte[] row : array) {
            if (row.length != array[0].length) {
                throw new IllegalArgumentException("Array shape is not a matrix. All rows must have the same length. " + row.length + " != " + array[0].length);
            }
        }
        long[] shape = new long[]{array.length, array[0].length};
        ITensor tensor = new Tensor(DataType.INT8, shape, getBackend());
        long[] index = new long[2];
        for (int i = 0; i < array.length; i++) {
            index[0] = i;
            for (int j = 0; j < array[i].length; j++) {
                index[1] = j;
                tensor.setByte(array[i][j], index);
            }
        }
        return tensor;
    }

    @NotNull
    public ITensor matrix(short @NotNull [][] array) {
        for (short[] row : array) {
            if (row.length != array[0].length) {
                throw new IllegalArgumentException("Array shape is not a matrix. All rows must have the same length. " + row.length + " != " + array[0].length);
            }
        }
        long[] shape = new long[]{array.length, array[0].length};
        ITensor tensor = new Tensor(DataType.INT16, shape, getBackend());
        long[] index = new long[2];
        for (int i = 0; i < array.length; i++) {
            index[0] = i;
            for (int j = 0; j < array[i].length; j++) {
                index[1] = j;
                tensor.setShort(array[i][j], index);
            }
        }
        return tensor;
    }

    @NotNull
    public ITensor matrix(int @NotNull [][] array) {
        for (int[] row : array) {
            if (row.length != array[0].length) {
                throw new IllegalArgumentException("Array shape is not a matrix. All rows must have the same length. " + row.length + " != " + array[0].length);
            }
        }
        long[] shape = new long[]{array.length, array[0].length};
        ITensor tensor = new Tensor(DataType.INT32, shape, getBackend());
        long[] index = new long[2];
        for (int i = 0; i < array.length; i++) {
            index[0] = i;
            for (int j = 0; j < array[i].length; j++) {
                index[1] = j;
                tensor.setInt(array[i][j], index);
            }
        }
        return tensor;
    }

    @NotNull
    public ITensor matrix(long @NotNull [][] array) {
        for (long[] row : array) {
            if (row.length != array[0].length) {
                throw new IllegalArgumentException("Array shape is not a matrix. All rows must have the same length. " + row.length + " != " + array[0].length);
            }
        }
        long[] shape = new long[]{array.length, array[0].length};
        ITensor tensor = new Tensor(DataType.INT64, shape, getBackend());
        long[] index = new long[2];
        for (int i = 0; i < array.length; i++) {
            index[0] = i;
            for (int j = 0; j < array[i].length; j++) {
                index[1] = j;
                tensor.setLong(array[i][j], index);
            }
        }
        return tensor;
    }

    @NotNull
    public ITensor matrix(float @NotNull [][] array) {
        for (float[] row : array) {
            if (row.length != array[0].length) {
                throw new IllegalArgumentException("Array shape is not a matrix. All rows must have the same length. " + row.length + " != " + array[0].length);
            }
        }
        long[] shape = new long[]{array.length, array[0].length};
        ITensor tensor = new Tensor(DataType.FLOAT32, shape, getBackend());
        long[] index = new long[2];
        for (int i = 0; i < array.length; i++) {
            index[0] = i;
            for (int j = 0; j < array[i].length; j++) {
                index[1] = j;
                tensor.setFloat(array[i][j], index);
            }
        }
        return tensor;
    }

    @NotNull
    public ITensor matrix(double @NotNull [][] array) {
        for (double[] row : array) {
            if (row.length != array[0].length) {
                throw new IllegalArgumentException("Array shape is not a matrix. All rows must have the same length. " + row.length + " != " + array[0].length);
            }
        }
        long[] shape = new long[]{array.length, array[0].length};
        ITensor tensor = new Tensor(DataType.FLOAT64, shape, getBackend());
        long[] index = new long[2];
        for (int i = 0; i < array.length; i++) {
            index[0] = i;
            for (int j = 0; j < array[i].length; j++) {
                index[1] = j;
                tensor.setDouble(array[i][j], index);
            }
        }
        return tensor;
    }

    @NotNull
    public ITensor ndarray(Object array) {
        Class<?> arrayClass = array.getClass();
        Class<?> componentClass = ArrayUtils.getComponentType(array);
        long[] arrayShape = ShapeUtils.getArrayShape(array);
        int nElements = Math.toIntExact(ShapeUtils.getNumElements(arrayShape));
        DataType dataType = DataType.fromClass(componentClass);
        Object elements = ArrayUtils.getElementsFlat(array);
        ITensor tensor = new Tensor(dataType, arrayShape, getBackend());
        // TODO: Make this more efficient by not having a second (flattened) copy of the array.
        if (elements instanceof byte[] byteArray) {
            for (int i = 0; i < nElements; i++) {
                tensor.setByteFlat((byteArray[i]), i);
            }
        } else if (elements instanceof short[] shortArray) {
            for (int i = 0; i < nElements; i++) {
                tensor.setShortFlat((shortArray[i]), i);
            }
        } else if (elements instanceof int[] intArray) {
            for (int i = 0; i < nElements; i++) {
                tensor.setIntFlat((intArray[i]), i);
            }
        } else if (elements instanceof long[] longArray) {
            for (int i = 0; i < nElements; i++) {
                tensor.setLongFlat((longArray[i]), i);
            }
        } else if (elements instanceof float[] floatArray) {
            for (int i = 0; i < nElements; i++) {
                tensor.setFloatFlat((floatArray[i]), i);
            }
        } else if (elements instanceof double[] doubleArray) {
            for (int i = 0; i < nElements; i++) {
                tensor.setDoubleFlat((doubleArray[i]), i);
            }
        } else {
            throw new IllegalArgumentException("Unsupported array type: " + arrayClass.getName());
        }
        return tensor;
    }

    public enum BackendType {

        JVM {
            @Override
            public SciCoreBackend newInstance() {
                return new JvmBackend();
            }
        };

        public abstract SciCoreBackend newInstance();
    }
}
