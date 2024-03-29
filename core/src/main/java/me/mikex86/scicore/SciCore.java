package me.mikex86.scicore;

import me.mikex86.scicore.backend.AbstractSciCoreBackend;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.backend.OperationRegistry;
import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.graph.*;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.memory.DirectMemoryManager;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.utils.ArrayUtils;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.nio.*;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class SciCore implements ISciCore {

    @NotNull
    private final DirectMemoryManager directMemoryManager = new DirectMemoryManager();

    @NotNull
    private final JvmBackend jvmBackend = new JvmBackend();

    @NotNull
    private ISciCoreBackend sciCoreBackend = jvmBackend;

    @NotNull
    private final OperationRegistry operationRegistry = new OperationRegistry();

    @NotNull
    private final IGraphRecorder operationRecorder = new GraphRecorder(operationRegistry);

    {
        jvmBackend.setOperationRecorder(operationRecorder);
        jvmBackend.setDirectMemoryManager(directMemoryManager);
        operationRegistry.pushLayer(jvmBackend); // JVM backend is always there as a fallback, if operations are not implemented in higher layers
    }

    @NotNull
    private final Random random = new Random();

    @Override
    @NotNull
    public ITensor zeros(@NotNull DataType dataType, long @NotNull ... shape) {
        ISciCoreBackend backend = getBackend();
        return backend.createTensor(dataType, shape);
    }

    @Override
    public void seed(long seed) {
        random.setSeed(seed);
    }

    @Override
    @NotNull
    public ITensor uniform(@NotNull DataType dataType, long @NotNull ... shape) {
        // TODO: CREATE FILL_UNIFORM OPERATION THAT CAN BE ACCELERATED
        ISciCoreBackend backend = getBackend();
        ITensor tensor = backend.createTensor(dataType, shape);
        long numberOfElements = tensor.getNumberOfElements();
        switch (dataType) {
            case FLOAT32 -> {
                DirectMemoryHandle handle = directMemoryManager.alloc(numberOfElements, DataType.FLOAT32);
                FloatBuffer buffer = handle.asFloatBuffer();
                for (int i = 0; i < numberOfElements; i++) {
                    buffer.put(random.nextFloat());
                }
                buffer.flip();
                tensor.setContents(buffer);
                handle.free();
            }
            case FLOAT64 -> {
                DirectMemoryHandle handle = directMemoryManager.alloc(numberOfElements, DataType.FLOAT64);
                DoubleBuffer buffer = handle.asDoubleBuffer();
                for (int i = 0; i < numberOfElements; i++) {
                    buffer.put(random.nextDouble());
                }
                buffer.flip();
                tensor.setContents(buffer);
                handle.free();
            }
            default -> throw new IllegalArgumentException("Unsupported data type: " + dataType);
        }
        return tensor;
    }

    @Override
    public @NotNull ITensor randint(@NotNull DataType dataType, long min, long max, long @NotNull ... shape) {
        // TODO: CREATE RAND_INT OPERATION THAT CAN BE ACCELERATED
        ISciCoreBackend backend = getBackend();
        ITensor tensor = backend.createTensor(dataType, shape);
        long numberOfElements = tensor.getNumberOfElements();
        switch (dataType) {
            case INT8 -> {
                DirectMemoryHandle handle = directMemoryManager.alloc(numberOfElements, DataType.INT8);
                ByteBuffer buffer = handle.asByteBuffer();
                for (int i = 0; i < numberOfElements; i++) {
                    buffer.put((byte) (random.nextInt((int) (max - min)) + min));
                }
                buffer.flip();
                tensor.setContents(buffer);
            }
            case INT16 -> {
                DirectMemoryHandle handle = directMemoryManager.alloc(numberOfElements, DataType.INT16);
                ShortBuffer buffer = handle.asShortBuffer();
                for (int i = 0; i < numberOfElements; i++) {
                    buffer.put((short) (random.nextInt((int) (max - min)) + min));
                }
                buffer.flip();
                tensor.setContents(buffer);
            }
            case INT32 -> {
                DirectMemoryHandle handle = directMemoryManager.alloc(numberOfElements, DataType.INT32);
                IntBuffer buffer = handle.asIntBuffer();
                for (int i = 0; i < numberOfElements; i++) {
                    buffer.put(random.nextInt((int) (max - min)) + (int) min);
                }
                tensor.setContents(buffer);
            }
            case INT64 -> {
                DirectMemoryHandle handle = directMemoryManager.alloc(numberOfElements, DataType.INT64);
                LongBuffer buffer = handle.asLongBuffer();
                for (int i = 0; i < numberOfElements; i++) {
                    buffer.put(random.nextLong() % (max - min) + min);
                }
                buffer.flip();
                tensor.setContents(buffer);
            }
        }
        return tensor;
    }

    @Override
    @NotNull
    public ITensor gaussian(@NotNull DataType dataType, long @NotNull ... shape) {
        // TODO: CREATE FILL_GAUSSIAN OPERATION THAT CAN BE ACCELERATED
        ISciCoreBackend backend = getBackend();
        ITensor tensor = backend.createTensor(dataType, shape);
        long numberOfElements = tensor.getNumberOfElements();
        switch (dataType) {
            case FLOAT32 -> {
                DirectMemoryHandle handle = directMemoryManager.alloc(numberOfElements, DataType.FLOAT32);
                FloatBuffer buffer = handle.asFloatBuffer();
                for (long i = 0; i < numberOfElements; i++) {
                    buffer.put((float) random.nextGaussian());
                }
                buffer.flip();
                tensor.setContents(buffer);
                handle.free();
            }
            case FLOAT64 -> {
                DirectMemoryHandle handle = directMemoryManager.alloc(numberOfElements, DataType.FLOAT64);
                DoubleBuffer buffer = handle.asDoubleBuffer();
                for (long i = 0; i < numberOfElements; i++) {
                    buffer.put(random.nextGaussian());
                }
                buffer.flip();
                tensor.setContents(buffer);
                handle.free();
            }
            default -> throw new IllegalArgumentException("Unsupported data type for gaussian: " + dataType);
        }
        return tensor;
    }

    @Override
    public void addBackend(@NotNull BackendType backendType) {
        if (backendType == BackendType.JVM) {
            return; // we already have the base JVM layer in #jvmBackend
        }
        sciCoreBackend = backendType.newInstance();
        if (!(sciCoreBackend instanceof AbstractSciCoreBackend abstractSciCoreBackend)) {
            throw new IllegalArgumentException("Backend does not implement AbstractSciCoreBackend: " + sciCoreBackend.getClass().getName());
        }
        abstractSciCoreBackend.setOperationRecorder(operationRecorder);
        abstractSciCoreBackend.setDirectMemoryManager(directMemoryManager);
        operationRegistry.pushLayer(sciCoreBackend);
    }

    @Override
    @NotNull
    public ISciCoreBackend getBackend() {
        return sciCoreBackend;
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
    public @NotNull ITensor matmul(@NotNull ITensor a, @NotNull ITensor b, boolean transposeA, boolean transposeB) {
        return a.matmul(b, transposeA, transposeB);
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
        ByteBuffer buffer = ByteBuffer.wrap(array);
        tensor.setContents(buffer);
        return tensor;
    }

    @Override
    @NotNull
    public ITensor array(short @NotNull [] array) {
        ISciCoreBackend backend = getBackend();
        ITensor tensor = backend.createTensor(DataType.INT16, new long[]{array.length});
        ShortBuffer buffer = ShortBuffer.wrap(array);
        tensor.setContents(buffer);
        return tensor;
    }

    @Override
    @NotNull
    public ITensor array(int @NotNull [] array) {
        ISciCoreBackend backend = getBackend();
        ITensor tensor = backend.createTensor(DataType.INT32, new long[]{array.length});
        IntBuffer buffer = IntBuffer.wrap(array);
        tensor.setContents(buffer);
        return tensor;
    }

    @Override
    @NotNull
    public ITensor array(long @NotNull [] array) {
        ISciCoreBackend backend = getBackend();
        ITensor tensor = backend.createTensor(DataType.INT64, new long[]{array.length});
        LongBuffer buffer = LongBuffer.wrap(array);
        tensor.setContents(buffer);
        return tensor;
    }

    @Override
    @NotNull
    public ITensor array(float @NotNull [] array) {
        ISciCoreBackend backend = getBackend();
        ITensor tensor = backend.createTensor(DataType.FLOAT32, new long[]{array.length});
        FloatBuffer buffer = FloatBuffer.wrap(array);
        tensor.setContents(buffer);
        return tensor;
    }

    @Override
    @NotNull
    public ITensor array(double @NotNull [] array) {
        ISciCoreBackend backend = getBackend();
        ITensor tensor = backend.createTensor(DataType.FLOAT64, new long[]{array.length});
        DoubleBuffer buffer = DoubleBuffer.wrap(array);
        tensor.setContents(buffer);
        return tensor;
    }

    @Override
    public @NotNull ITensor array(boolean @NotNull [] array) {
        ISciCoreBackend backend = getBackend();
        ITensor tensor = backend.createTensor(DataType.BOOLEAN, new long[]{array.length});
        tensor.setContents(array);
        return tensor;
    }

    // TODO: OPTIMIZE MATRIX CONSTRUCTION BY BACKING THE TENSOR WITH THE ALLOCATED MEMORY, AS OPPOSED TO AN UNNECESSARY COPY

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
        int nElements = Math.toIntExact(ShapeUtils.getNumElements(shape));
        ITensor tensor = backend.createTensor(DataType.INT8, shape);
        DirectMemoryHandle memoryHandle = directMemoryManager.alloc(nElements, DataType.INT8);
        ByteBuffer buffer = memoryHandle.asByteBuffer();
        for (byte[] row : array) {
            buffer.put(row);
        }
        buffer.flip();
        tensor.setContents(buffer);
        memoryHandle.free();
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
        int nElements = Math.toIntExact(ShapeUtils.getNumElements(shape));
        ITensor tensor = backend.createTensor(DataType.INT16, shape);
        DirectMemoryHandle memoryHandle = directMemoryManager.alloc(nElements, DataType.INT16);
        ShortBuffer buffer = memoryHandle.asShortBuffer();
        for (short[] row : array) {
            buffer.put(row);
        }
        buffer.flip();
        tensor.setContents(buffer);
        memoryHandle.free();
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
        int nElements = Math.toIntExact(ShapeUtils.getNumElements(shape));
        ITensor tensor = backend.createTensor(DataType.INT32, shape);
        DirectMemoryHandle memoryHandle = directMemoryManager.alloc(nElements, DataType.INT32);
        IntBuffer buffer = memoryHandle.asIntBuffer();
        for (int[] row : array) {
            buffer.put(row);
        }
        buffer.flip();
        tensor.setContents(buffer);
        memoryHandle.free();
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
        int nElements = Math.toIntExact(ShapeUtils.getNumElements(shape));
        ITensor tensor = backend.createTensor(DataType.INT64, shape);
        DirectMemoryHandle memoryHandle = directMemoryManager.alloc(nElements, DataType.INT64);
        LongBuffer buffer = memoryHandle.asLongBuffer();
        for (long[] row : array) {
            buffer.put(row);
        }
        buffer.flip();
        tensor.setContents(buffer);
        memoryHandle.free();
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
        int nElements = Math.toIntExact(ShapeUtils.getNumElements(shape));
        ITensor tensor = backend.createTensor(DataType.FLOAT32, shape);
        DirectMemoryHandle memoryHandle = directMemoryManager.alloc(nElements, DataType.FLOAT32);
        FloatBuffer buffer = memoryHandle.asFloatBuffer();
        for (float[] row : array) {
            buffer.put(row);
        }
        buffer.flip();
        tensor.setContents(buffer);
        memoryHandle.free();
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
        int nElements = Math.toIntExact(ShapeUtils.getNumElements(shape));
        ITensor tensor = backend.createTensor(DataType.FLOAT64, shape);
        DirectMemoryHandle memoryHandle = directMemoryManager.alloc(nElements, DataType.FLOAT64);
        DoubleBuffer buffer = memoryHandle.asDoubleBuffer();
        for (double[] row : array) {
            buffer.put(row);
        }
        buffer.flip();
        tensor.setContents(buffer);
        memoryHandle.free();
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
        int nElements = Math.toIntExact(ShapeUtils.getNumElements(shape));
        ITensor tensor = backend.createTensor(DataType.BOOLEAN, shape);
        boolean[] data = new boolean[nElements];
        int i = 0;
        for (boolean[] row : array) {
            System.arraycopy(row, 0, data, i, row.length);
            i += row.length;
        }
        tensor.setContents(data);
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
        } else if (elements instanceof boolean[] booleanArray) {
            for (int i = 0; i < nElements; i++) {
                tensor.setBooleanFlat((booleanArray[i]), i);
            }
        } else {
            throw new IllegalArgumentException("Unsupported array type: " + arrayClass.getName());
        }
        return tensor;
    }

    @Override
    public @NotNull ITensor zerosLike(@NotNull ITensor reference) {
        ISciCoreBackend backend = getBackend();
        DataType dataType = reference.getDataType();
        long[] shape = reference.getShape();
        return backend.createTensor(dataType, shape);
    }

    @Override
    @NotNull
    public ITensor onesLike(@NotNull ITensor reference) {
        ITensor tensor = zerosLike(reference);
        tensor.fill(1);
        return tensor;
    }

    @Override
    @NotNull
    public IGraph getExecutionGraphUpTo(@NotNull ITensor root) {
        return operationRecorder.getExecutionGraphTo(getBackend(), root);
    }

    @Override
    public @NotNull IGraph getBackpropagationGraphUpTo(@NotNull ITensor root, @NotNull List<ITensor> parameters) {
        return operationRecorder.getBackpropagationGraphTo(getBackend(), root, parameters);
    }

    @Override
    public @NotNull ITensor scalar(byte value) {
        ISciCoreBackend backend = getBackend();
        ITensor tensor = backend.createTensor(DataType.INT8, new long[0]);
        tensor.setByteFlat(value, 0);
        return tensor;
    }

    @Override
    public @NotNull ITensor scalar(short value) {
        ISciCoreBackend backend = getBackend();
        ITensor tensor = backend.createTensor(DataType.INT16, new long[0]);
        tensor.setShortFlat(value, 0);
        return tensor;
    }

    @Override
    public @NotNull ITensor scalar(int value) {
        ISciCoreBackend backend = getBackend();
        ITensor tensor = backend.createTensor(DataType.INT32, new long[0]);
        tensor.setIntFlat(value, 0);
        return tensor;
    }

    @Override
    public @NotNull ITensor scalar(long value) {
        ISciCoreBackend backend = getBackend();
        ITensor tensor = backend.createTensor(DataType.INT64, new long[0]);
        tensor.setLongFlat(value, 0);
        return tensor;
    }

    @Override
    public @NotNull ITensor scalar(float value) {
        ISciCoreBackend backend = getBackend();
        long[] shape = new long[]{1};
        ITensor tensor = backend.createTensor(DataType.FLOAT32, new long[0]);
        tensor.setFloatFlat(value, 0);
        return tensor;
    }

    @Override
    public @NotNull ITensor scalar(double value) {
        ISciCoreBackend backend = getBackend();
        ITensor tensor = backend.createTensor(DataType.FLOAT64, new long[0]);
        tensor.setDoubleFlat(value, 0);
        return tensor;
    }

    @Override
    public @NotNull ITensor scalar(boolean value) {
        ISciCoreBackend backend = getBackend();
        ITensor tensor = backend.createTensor(DataType.BOOLEAN, new long[0]);
        tensor.setBooleanFlat(value, 0);
        return tensor;
    }

    @Override
    public @NotNull ITensor pow(ITensor base, byte exponent) {
        return base.pow(exponent);
    }

    @Override
    public @NotNull ITensor pow(ITensor base, short exponent) {
        return base.pow(exponent);
    }

    @Override
    public @NotNull ITensor pow(ITensor base, int exponent) {
        return base.pow(exponent);
    }

    @Override
    public @NotNull ITensor pow(ITensor base, long exponent) {
        return base.pow(exponent);
    }

    @Override
    public @NotNull ITensor pow(ITensor base, float exponent) {
        return base.pow(exponent);
    }

    @Override
    public @NotNull ITensor pow(ITensor base, double exponent) {
        return base.pow(exponent);
    }

    @Override
    public @NotNull ITensor arange(long start, long stop, long step, @NotNull DataType dataType) {
        ISciCoreBackend backend = getBackend();
        long nElements = (stop - start) / step;
        long[] shape = new long[]{nElements};
        ITensor tensor = backend.createTensor(dataType, shape);
        for (long i = 0; i < nElements; i++) {
            tensor.setByLongFlat(start, i);
            start += step;
        }
        return tensor;
    }

    @Override
    public @NotNull ITensor arange(double start, double stop, double step, @NotNull DataType dataType) {
        ISciCoreBackend backend = getBackend();
        long nElements = (long) Math.ceil((stop - start) / step);
        long[] shape = new long[]{nElements};
        ITensor tensor = backend.createTensor(dataType, shape);
        for (long i = 0; i < nElements; i++) {
            tensor.setByDoubleFlat(start, i);
            start += step;
        }
        return tensor;
    }

    @NotNull
    public ITensor stack(int dimension, @NotNull ITensor @NotNull ... tensors) {
        return operationRecorder.recordOperation(OperationType.STACK, OptionBundle.of(sciCoreBackend, Map.of(
                "dimension", scalar(dimension)
        )), tensors);
    }

    @Override
    public @NotNull ITensor crossEntropy(@NotNull ITensor logits, @NotNull ITensor target, @Nullable Long ignoreIndex) {
        try (ITensor counts = logits.exp(); ITensor totalCounts = counts.reduceSum(1, true);
             ITensor probabilities = counts.divide(totalCounts);
             ITensor allBatchesIdx = arange(0, logits.getShape()[0], 1, DataType.INT64)) {
            ITensor targetIndices = target;
            ITensor mask = null;
            if (ignoreIndex != null) {
                mask = targetIndices.where(ignoreIndex, 0, 1);
                targetIndices = targetIndices.multiply(mask);
            }
            try (ITensor probabilitiesAssignedToCorrectLabels = probabilities.get(allBatchesIdx, targetIndices)) {
                if (targetIndices != target) {
                    targetIndices.close();
                }
                ITensor probabilitiesAssignedToCorrectLabelsMasked = probabilitiesAssignedToCorrectLabels;
                if (mask != null) {
                    ITensor tmp = probabilitiesAssignedToCorrectLabels.multiply(mask);
                    try (ITensor negativeMask = mask.leftMinus(1f)) {
                        // TODO: FIX THIS
                        probabilitiesAssignedToCorrectLabelsMasked = tmp.plus(negativeMask);
                    }
                    tmp.close();
                    mask.close();
                }
                try (ITensor logProbabilitiesAssignedToCorrectLabels = probabilitiesAssignedToCorrectLabelsMasked.log()) {

                    if (probabilitiesAssignedToCorrectLabelsMasked != probabilitiesAssignedToCorrectLabels) {
                        probabilitiesAssignedToCorrectLabelsMasked.close();
                    }
                    try (ITensor meanLogProbabilitiesAssignedToCorrectLabels = logProbabilitiesAssignedToCorrectLabels.mean(-1, false)) {
                        return meanLogProbabilitiesAssignedToCorrectLabels.multiply(-1f);
                    }
                }
            }
        }
    }

    @NotNull
    @Override
    public ITensor multinomial(@NotNull ITensor probabilities, long numSamples) {
        long[] resultShape = new long[probabilities.getShape().length];
        System.arraycopy(probabilities.getShape(), 0, resultShape, 0, probabilities.getShape().length);
        resultShape[resultShape.length - 1] = numSamples;
        ITensor result = zeros(DataType.INT64, resultShape);
        long[] probabilitiesShape = probabilities.getShape();
        long numProbabilitiesPerDistribution = probabilitiesShape[probabilitiesShape.length - 1];
        ITensor probabilitiesForEachDistribution = probabilities.view(-1, numProbabilitiesPerDistribution);
        long[] shape = probabilitiesForEachDistribution.getShape();
        long numProbabilityDistributions = shape[0];
        for (long i = 0; i < numProbabilityDistributions; i++) {
            ITensor probabilitiesForOneDistribution = probabilitiesForEachDistribution.getView(i);
            ITensor resultForOneDistribution = result.getView(i);
            for (int j = 0; j < numSamples; j++) {
                double cumulativeProbability = 0;
                double randomValue = random.nextDouble();
                for (long k = 0; k < numProbabilitiesPerDistribution; k++) {
                    cumulativeProbability += probabilitiesForOneDistribution.getAsDouble(k);
                    if (cumulativeProbability >= randomValue) {
                        resultForOneDistribution.setLong(k, j);
                        break;
                    }
                }
            }
        }
        return result;
    }

    @Override
    public void disableBackendFallback() {
        operationRegistry.disableFallthrough();
    }

    @Override
    public @NotNull ITensor triangle(@NotNull DataType dataType, long dim0, long dim1, double topValue, double bottomValue) {
        ITensor result = zeros(dataType, dim0, dim1);
        IGraphRecorder graphRecorder = sciCoreBackend.getOperationRecorder();
        graphRecorder.recordOperation(OperationType.FILL_TRIANGLE, OptionBundle.of(sciCoreBackend, Map.of(
                "topValue", scalar(topValue),
                "bottomValue", scalar(bottomValue)
        )), result);
        return result;
    }

    private boolean isTraining = true;

    @Override
    public boolean isTraining() {
        return isTraining;
    }

    @Override
    public void eval() {
        isTraining = false;
    }

    @Override
    public void train() {
        isTraining = true;
    }

}
