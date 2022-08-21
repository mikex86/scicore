package me.mikex86.scicore;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.utils.ArrayUtils;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.Objects;
import java.util.Random;

public class SciCore implements ISciCore {

    @Nullable
    private ISciCoreBackend sciCoreBackend = null;

    @Override
    @NotNull
    public ITensor zeros(@NotNull DataType dataType, long @NotNull ... shape) {
        ISciCoreBackend backend = getBackend();
        return backend.createTensor(dataType, shape);
    }

    @Override
    @NotNull
    public ITensor uniform(@NotNull DataType dataType, long @NotNull ... shape) {
        // TODO: CREATE FILL_UNIFORM OPERATION THAT CAN BE ACCELERATED
        ISciCoreBackend backend = getBackend();
        ITensor tensor = backend.createTensor(dataType, shape);
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

    @Override
    public void setBackend(@NotNull BackendType backendType) {
        if (sciCoreBackend != null) {
            throw new IllegalStateException("SciCore backend already initialized!");
        }
        sciCoreBackend = backendType.newInstance();
    }

    @Override
    @NotNull
    public ISciCoreBackend getBackend() {
        return Objects.requireNonNull(sciCoreBackend, "Backend has not yet been configured");
    }

    @Override
    public void fill(@NotNull ITensor tensor, byte i) {
        tensor.fill(i);
    }

    @Override
    public void fill(@NotNull ITensor tensor, short i) {
        tensor.fill(i);
    }

    @Override
    public void fill(@NotNull ITensor tensor, int i) {
        tensor.fill(i);
    }

    @Override
    public void fill(@NotNull ITensor tensor, long value) {
        tensor.fill(value);
    }

    @Override
    public void fill(@NotNull ITensor tensor, float f) {
        tensor.fill(f);
    }

    @Override
    public void fill(@NotNull ITensor tensor, double d) {
        tensor.fill(d);
    }

    @Override
    public void fill(@NotNull ITensor tensor, boolean value) {
        tensor.fill(value);
    }

    @Override
    @NotNull
    public ITensor matmul(@NotNull ITensor a, @NotNull ITensor b) {
        return a.matmul(b);
    }

    @Override
    @NotNull
    public ITensor exp(@NotNull ITensor a) {
        return a.exp();
    }

    @Override
    @NotNull
    public ITensor softmax(@NotNull ITensor input, int dimension) {
        return input.softmax(dimension);
    }

    @Override
    @NotNull
    public ITensor reduceSum(@NotNull ITensor input, int dimension) {
        return input.reduceSum(dimension);
    }

    @Override
    @NotNull
    public ITensor array(byte @NotNull [] array) {
        ISciCoreBackend backend = getBackend();
        ITensor tensor = backend.createTensor(DataType.INT8, new long[]{array.length});
        for (int i = 0; i < array.length; i++) {
            tensor.setByteFlat(array[i], i);
        }
        return tensor;
    }

    @Override
    @NotNull
    public ITensor array(short @NotNull [] array) {
        ISciCoreBackend backend = getBackend();
        ITensor tensor = backend.createTensor(DataType.INT16, new long[]{array.length});
        for (int i = 0; i < array.length; i++) {
            tensor.setShortFlat(array[i], i);
        }
        return tensor;
    }

    @Override
    @NotNull
    public ITensor array(int @NotNull [] array) {
        ISciCoreBackend backend = getBackend();
        ITensor tensor = backend.createTensor(DataType.INT32, new long[]{array.length});
        for (int i = 0; i < array.length; i++) {
            tensor.setIntFlat(array[i], i);
        }
        return tensor;
    }

    @Override
    @NotNull
    public ITensor array(long @NotNull [] array) {
        ISciCoreBackend backend = getBackend();
        ITensor tensor = backend.createTensor(DataType.INT64, new long[]{array.length});
        for (int i = 0; i < array.length; i++) {
            tensor.setLongFlat(array[i], i);
        }
        return tensor;
    }

    @Override
    @NotNull
    public ITensor array(float @NotNull [] array) {
        ISciCoreBackend backend = getBackend();
        ITensor tensor = backend.createTensor(DataType.FLOAT32, new long[]{array.length});
        for (int i = 0; i < array.length; i++) {
            tensor.setFloatFlat(array[i], i);
        }
        return tensor;
    }

    @Override
    @NotNull
    public ITensor array(double @NotNull [] array) {
        ISciCoreBackend backend = getBackend();
        ITensor tensor = backend.createTensor(DataType.FLOAT64, new long[]{array.length});
        for (int i = 0; i < array.length; i++) {
            tensor.setDoubleFlat(array[i], i);
        }
        return tensor;
    }

    @Override
    public @NotNull ITensor array(boolean @NotNull [] array) {
        ISciCoreBackend backend = getBackend();
        ITensor tensor = backend.createTensor(DataType.BOOLEAN, new long[]{array.length});
        for (int i = 0; i < array.length; i++) {
            tensor.setBooleanFlat(array[i], i);
        }
        return tensor;
    }

    @Override
    @NotNull
    public ITensor matrix(byte @NotNull [][] array) {
        ISciCoreBackend backend = getBackend();
        for (byte[] row : array) {
            if (row.length != array[0].length) {
                throw new IllegalArgumentException("Array shape is not a matrix. All rows must have the same length. " + row.length + " != " + array[0].length);
            }
        }
        long[] shape = new long[]{array.length, array[0].length};
        ITensor tensor = backend.createTensor(DataType.INT8, shape);
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

    @Override
    @NotNull
    public ITensor matrix(short @NotNull [][] array) {
        ISciCoreBackend backend = getBackend();
        for (short[] row : array) {
            if (row.length != array[0].length) {
                throw new IllegalArgumentException("Array shape is not a matrix. All rows must have the same length. " + row.length + " != " + array[0].length);
            }
        }
        long[] shape = new long[]{array.length, array[0].length};
        ITensor tensor = backend.createTensor(DataType.INT16, shape);
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

    @Override
    @NotNull
    public ITensor matrix(int @NotNull [][] array) {
        ISciCoreBackend backend = getBackend();
        for (int[] row : array) {
            if (row.length != array[0].length) {
                throw new IllegalArgumentException("Array shape is not a matrix. All rows must have the same length. " + row.length + " != " + array[0].length);
            }
        }
        long[] shape = new long[]{array.length, array[0].length};
        ITensor tensor = backend.createTensor(DataType.INT32, shape);
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

    @Override
    @NotNull
    public ITensor matrix(long @NotNull [][] array) {
        ISciCoreBackend backend = getBackend();
        for (long[] row : array) {
            if (row.length != array[0].length) {
                throw new IllegalArgumentException("Array shape is not a matrix. All rows must have the same length. " + row.length + " != " + array[0].length);
            }
        }
        long[] shape = new long[]{array.length, array[0].length};
        ITensor tensor = backend.createTensor(DataType.INT64, shape);
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

    @Override
    @NotNull
    public ITensor matrix(float @NotNull [][] array) {
        ISciCoreBackend backend = getBackend();
        for (float[] row : array) {
            if (row.length != array[0].length) {
                throw new IllegalArgumentException("Array shape is not a matrix. All rows must have the same length. " + row.length + " != " + array[0].length);
            }
        }
        long[] shape = new long[]{array.length, array[0].length};
        ITensor tensor = backend.createTensor(DataType.FLOAT32, shape);
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

    @Override
    @NotNull
    public ITensor matrix(double @NotNull [][] array) {
        ISciCoreBackend backend = getBackend();
        for (double[] row : array) {
            if (row.length != array[0].length) {
                throw new IllegalArgumentException("Array shape is not a matrix. All rows must have the same length. " + row.length + " != " + array[0].length);
            }
        }
        long[] shape = new long[]{array.length, array[0].length};
        ITensor tensor = backend.createTensor(DataType.FLOAT64, shape);
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

    @Override
    public @NotNull ITensor matrix(boolean @NotNull [][] array) {
        ISciCoreBackend backend = getBackend();
        for (boolean[] row : array) {
            if (row.length != array[0].length) {
                throw new IllegalArgumentException("Array shape is not a matrix. All rows must have the same length. " + row.length + " != " + array[0].length);
            }
        }
        long[] shape = new long[]{array.length, array[0].length};
        ITensor tensor = backend.createTensor(DataType.BOOLEAN, shape);
        long[] index = new long[2];
        for (int i = 0; i < array.length; i++) {
            index[0] = i;
            for (int j = 0; j < array[i].length; j++) {
                index[1] = j;
                tensor.setBoolean(array[i][j], index);
            }
        }
        return tensor;
    }

    @Override
    @NotNull
    public ITensor ndarray(Object array) {
        ISciCoreBackend backend = getBackend();
        Class<?> arrayClass = array.getClass();
        Class<?> componentClass = ArrayUtils.getComponentType(array);
        long[] arrayShape = ShapeUtils.getArrayShape(array);
        int nElements = Math.toIntExact(ShapeUtils.getNumElements(arrayShape));
        DataType dataType = DataType.fromClass(componentClass);
        Object elements = ArrayUtils.getElementsFlat(array);
        ITensor tensor = backend.createTensor(dataType, arrayShape);
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

    @Override
    @NotNull
    public ITensor onesLike(@NotNull ITensor reference) {
        ISciCoreBackend backend = getBackend();
        DataType dataType = reference.getDataType();
        long[] shape = reference.getShape();
        ITensor tensor = backend.createTensor(dataType, shape);
        tensor.fill(1);
        return tensor;
    }

    @Override
    @NotNull
    public IGraph getGraphUpTo(@NotNull ITensor tensor) {
        return getBackend().getOperationRecorder().getGraphFor(tensor);
    }

    @Override
    public @NotNull ITensor scalar(byte value) {
        ISciCoreBackend backend = getBackend();
        long[] shape = new long[]{1};
        ITensor tensor = backend.createTensor(DataType.INT8, shape);
        tensor.setByteFlat(value, 0);
        return tensor;
    }

    @Override
    public @NotNull ITensor scalar(short value) {
        ISciCoreBackend backend = getBackend();
        long[] shape = new long[]{1};
        ITensor tensor = backend.createTensor(DataType.INT16, shape);
        tensor.setShortFlat(value, 0);
        return tensor;
    }

    @Override
    public @NotNull ITensor scalar(int value) {
        ISciCoreBackend backend = getBackend();
        long[] shape = new long[]{1};
        ITensor tensor = backend.createTensor(DataType.INT32, shape);
        tensor.setIntFlat(value, 0);
        return tensor;
    }

    @Override
    public @NotNull ITensor scalar(long value) {
        ISciCoreBackend backend = getBackend();
        long[] shape = new long[]{1};
        ITensor tensor = backend.createTensor(DataType.INT64, shape);
        tensor.setLongFlat(value, 0);
        return tensor;
    }

    @Override
    public @NotNull ITensor scalar(float value) {
        ISciCoreBackend backend = getBackend();
        long[] shape = new long[]{1};
        ITensor tensor = backend.createTensor(DataType.FLOAT32, shape);
        tensor.setFloatFlat(value, 0);
        return tensor;
    }

    @Override
    public @NotNull ITensor scalar(double value) {
        ISciCoreBackend backend = getBackend();
        long[] shape = new long[]{1};
        ITensor tensor = backend.createTensor(DataType.FLOAT64, shape);
        tensor.setDoubleFlat(value, 0);
        return tensor;
    }

    @Override
    public @NotNull ITensor scalar(boolean value) {
        ISciCoreBackend backend = getBackend();
        long[] shape = new long[]{1};
        ITensor tensor = backend.createTensor(DataType.BOOLEAN, shape);
        tensor.setBooleanFlat(value, 0);
        return tensor;
    }
}
