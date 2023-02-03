package me.mikex86.scicore.tensor;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.IGraphRecorder;
import me.mikex86.scicore.graph.OperationType;
import me.mikex86.scicore.graph.OptionBundle;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Map;

import static me.mikex86.scicore.utils.StringUtils.formatFloat;

public abstract class AbstractTensor implements ITensor {

    // All implementations must set this variable!
    protected long numElements;

    private static long nextId = 0;

    private final long id;
    private boolean disposed = false;


    {
        synchronized (AbstractTensor.class) {
            id = nextId++;
        }
    }

    @Override
    public long getId() {
        return id;
    }

    @Nullable
    protected IGraph.ITensorNode associatedGraphNode;

    @Override
    public void setAssociatedGraphNode(@Nullable IGraph.ITensorNode graphNode) {
        this.associatedGraphNode = graphNode;
    }

    @Nullable
    public IGraph.ITensorNode getAssociatedGraphNode() {
        return associatedGraphNode;
    }


    @Override
    public void writeTo(@NotNull OutputStream outputStream) throws IOException {
        DirectMemoryHandle contents = getContentsAsDirectMemory();
        ByteBuffer byteBuffer = contents.asByteBuffer().order(ByteOrder.BIG_ENDIAN);
        byte[] copyBuffer = new byte[33554432]; //32MB
        while (byteBuffer.hasRemaining()) {
            int toCopy = Math.min(byteBuffer.remaining(), copyBuffer.length);
            byteBuffer.get(copyBuffer, 0, toCopy);
            outputStream.write(copyBuffer, 0, toCopy);
        }
    }

    @Override
    @NotNull
    public ITensor getView(long @NotNull ... indices) {
        long[] shape = getShape();
        validateIndices(indices);
        long[] strides = getStrides();

        long[] sliceShape = Arrays.copyOfRange(shape, indices.length, shape.length);
        long[] sliceStrides = ShapeUtils.makeStrides(sliceShape);

        long offset = ShapeUtils.getFlatIndex(indices, shape, strides);
        return new View(getSciCoreBackend(), getDataContainer(), sliceShape, offset, sliceStrides);
    }

    @Override
    public long getNumberOfElements() {
        return this.numElements;
    }

    @Override
    public byte getByte(long @NotNull ... indices) {
        return getByteFlat(ShapeUtils.getFlatIndex(indices, getShape(), getStrides()));
    }

    @Override
    public void setByte(byte value, long @NotNull ... indices) {
        setByteFlat(value, ShapeUtils.getFlatIndex(indices, getShape(), getStrides()));
    }

    @Override
    public short getShort(long @NotNull ... indices) {
        return getShortFlat(ShapeUtils.getFlatIndex(indices, getShape(), getStrides()));
    }

    @Override
    public void setShort(short value, long @NotNull ... indices) {
        setShortFlat(value, ShapeUtils.getFlatIndex(indices, getShape(), getStrides()));
    }

    @Override
    public int getInt(long @NotNull ... indices) {
        return getIntFlat(ShapeUtils.getFlatIndex(indices, getShape(), getStrides()));
    }

    @Override
    public void setInt(int value, long @NotNull ... indices) {
        setIntFlat(value, ShapeUtils.getFlatIndex(indices, getShape(), getStrides()));
    }

    @Override
    public void setLong(long value, long @NotNull ... indices) {
        setLongFlat(value, ShapeUtils.getFlatIndex(indices, getShape(), getStrides()));
    }

    @Override
    public long getLong(long @NotNull ... indices) {
        return getLongFlat(ShapeUtils.getFlatIndex(indices, getShape(), getStrides()));
    }

    @Override
    public float getFloat(long @NotNull ... indices) {
        return getFloatFlat(ShapeUtils.getFlatIndex(indices, getShape(), getStrides()));
    }

    @Override
    public void setFloat(float value, long @NotNull ... indices) {
        setFloatFlat(value, ShapeUtils.getFlatIndex(indices, getShape(), getStrides()));
    }

    @Override
    public double getDouble(long @NotNull ... indices) {
        return getDoubleFlat(ShapeUtils.getFlatIndex(indices, getShape(), getStrides()));
    }

    @Override
    public void setDouble(double value, long @NotNull ... indices) {
        setDoubleFlat(value, ShapeUtils.getFlatIndex(indices, getShape(), getStrides()));
    }

    @Override
    public boolean getBoolean(long @NotNull ... indices) {
        return getBooleanFlat(ShapeUtils.getFlatIndex(indices, getShape(), getStrides()));
    }

    @Override
    public void setBoolean(boolean value, long @NotNull ... indices) {
        setBooleanFlat(value, ShapeUtils.getFlatIndex(indices, getShape(), getStrides()));
    }

    @Override
    @NotNull
    public ITensor matmul(@NotNull ITensor other) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.MATMUL, backend, this, other);
    }

    @Override
    public @NotNull ITensor matmul(@NotNull ITensor other, boolean transposeSelf, boolean transposeOther) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor transposeSelfTensor = backend.createTensor(DataType.BOOLEAN, new long[0]);
             ITensor transposeOtherTensor = backend.createTensor(DataType.BOOLEAN, new long[0])) {
            transposeSelfTensor.setBooleanFlat(transposeSelf, 0);
            transposeOtherTensor.setBooleanFlat(transposeOther, 0);
            return operationRecorder.recordOperation(OperationType.MATMUL, OptionBundle.of(
                    backend,
                    Map.of(
                            "transposeA", transposeSelfTensor,
                            "transposeB", transposeOtherTensor
                    )
            ), this, other);
        }
    }

    @Override
    @NotNull
    public ITensor divide(@NotNull ITensor other) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.DIVIDE, backend, this, other);
    }

    @Override
    public @NotNull ITensor divide(byte value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT8, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.DIVIDE, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor divide(short value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT16, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.DIVIDE, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor divide(int value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT32, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.DIVIDE, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor divide(long value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT64, new long[0])) {
            valueScalar.setLongFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.DIVIDE, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor divide(float value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.FLOAT32, new long[0])) {
            valueScalar.setFloatFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.DIVIDE, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor divide(double value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.FLOAT64, new long[0])) {
            valueScalar.setDoubleFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.DIVIDE, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor leftDivide(@NotNull ITensor other) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.DIVIDE, backend, other, this);
    }

    @Override
    public @NotNull ITensor leftDivide(byte value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT8, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.DIVIDE, backend, valueScalar, this);
        }
    }

    @Override
    public @NotNull ITensor leftDivide(short value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT16, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.DIVIDE, backend, valueScalar, this);
        }
    }

    @Override
    public @NotNull ITensor leftDivide(int value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT32, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.DIVIDE, backend, valueScalar, this);
        }
    }

    @Override
    public @NotNull ITensor leftDivide(long value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT64, new long[0])) {
            valueScalar.setLongFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.DIVIDE, backend, valueScalar, this);
        }
    }

    @Override
    public @NotNull ITensor leftDivide(float value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.FLOAT32, new long[0])) {
            valueScalar.setFloatFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.DIVIDE, backend, valueScalar, this);
        }
    }

    @Override
    public @NotNull ITensor leftDivide(double value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.FLOAT64, new long[0])) {
            valueScalar.setDoubleFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.DIVIDE, backend, valueScalar, this);
        }
    }

    @Override
    public @NotNull ITensor plus(byte value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT8, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.PLUS, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor plus(short value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT16, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.PLUS, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor plus(int value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT32, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.PLUS, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor plus(long value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT64, new long[0])) {
            valueScalar.setLongFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.PLUS, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor plus(float value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.FLOAT32, new long[0])) {
            valueScalar.setFloatFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.PLUS, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor plus(double value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.FLOAT64, new long[0])) {
            valueScalar.setDoubleFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.PLUS, backend, this, valueScalar);
        }
    }

    @Override
    @NotNull
    public ITensor plus(@NotNull ITensor other) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.PLUS, backend, this, other);
    }

    @Override
    public void add(byte value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT8, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            operationRecorder.recordOperation(OperationType.PLUS_INPLACE, backend, this, valueScalar);
        }
    }

    @Override
    public void add(short value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT16, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            operationRecorder.recordOperation(OperationType.PLUS_INPLACE, backend, this, valueScalar);
        }
    }

    @Override
    public void add(int value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT32, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            operationRecorder.recordOperation(OperationType.PLUS_INPLACE, backend, this, valueScalar);
        }
    }

    @Override
    public void add(long value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();

        try (ITensor valueScalar = backend.createTensor(DataType.INT64, new long[0])) {
            valueScalar.setLongFlat(value, 0);
            operationRecorder.recordOperation(OperationType.PLUS_INPLACE, backend, this, valueScalar);
        }
    }

    @Override
    public void add(float value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.FLOAT32, new long[0])) {
            valueScalar.setFloatFlat(value, 0);
            operationRecorder.recordOperation(OperationType.PLUS_INPLACE, backend, this, valueScalar);
        }
    }

    @Override
    public void add(double value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.FLOAT64, new long[0])) {
            valueScalar.setDoubleFlat(value, 0);
            operationRecorder.recordOperation(OperationType.PLUS_INPLACE, backend, this, valueScalar);
        }
    }

    @Override
    public void add(@NotNull ITensor other) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        operationRecorder.recordOperation(OperationType.PLUS_INPLACE, backend, this, other);
    }

    @Override
    public @NotNull ITensor minus(byte value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT8, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.MINUS, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor minus(short value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT16, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.MINUS, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor minus(int value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT32, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.MINUS, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor minus(long value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT64, new long[0])) {
            valueScalar.setLongFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.MINUS, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor minus(float value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.FLOAT32, new long[0])) {
            valueScalar.setFloatFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.MINUS, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor minus(double value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.FLOAT64, new long[0])) {
            valueScalar.setDoubleFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.MINUS, backend, this, valueScalar);
        }
    }

    @Override
    @NotNull
    public ITensor minus(@NotNull ITensor other) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.MINUS, backend, this, other);
    }

    @Override
    public @NotNull ITensor leftMinus(byte value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT8, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.MINUS, backend, valueScalar, this);
        }
    }

    @Override
    public @NotNull ITensor leftMinus(short value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT16, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.MINUS, backend, valueScalar, this);
        }
    }

    @Override
    public @NotNull ITensor leftMinus(int value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT32, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.MINUS, backend, valueScalar, this);
        }
    }

    @Override
    public @NotNull ITensor leftMinus(long value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT64, new long[0])) {
            valueScalar.setLongFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.MINUS, backend, valueScalar, this);
        }
    }

    @Override
    public @NotNull ITensor leftMinus(float value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.FLOAT32, new long[0])) {
            valueScalar.setFloatFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.MINUS, backend, valueScalar, this);
        }
    }

    @Override
    public @NotNull ITensor leftMinus(double value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.FLOAT64, new long[0])) {
            valueScalar.setDoubleFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.MINUS, backend, valueScalar, this);
        }
    }

    @Override
    public void subtract(byte value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT8, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            operationRecorder.recordOperation(OperationType.MINUS_INPLACE, backend, this, valueScalar);
        }
    }

    @Override
    public void subtract(short value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT16, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            operationRecorder.recordOperation(OperationType.MINUS_INPLACE, backend, this, valueScalar);
        }
    }

    @Override
    public void subtract(int value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT32, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            operationRecorder.recordOperation(OperationType.MINUS_INPLACE, backend, this, valueScalar);
        }
    }

    @Override
    public void subtract(long value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT64, new long[0])) {
            valueScalar.setLongFlat(value, 0);
            operationRecorder.recordOperation(OperationType.MINUS_INPLACE, backend, this, valueScalar);
        }
    }

    @Override
    public void subtract(float value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.FLOAT32, new long[0])) {
            valueScalar.setFloatFlat(value, 0);
            operationRecorder.recordOperation(OperationType.MINUS_INPLACE, backend, this, valueScalar);
        }
    }

    @Override
    public void subtract(double value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.FLOAT64, new long[0])) {
            valueScalar.setDoubleFlat(value, 0);
            operationRecorder.recordOperation(OperationType.MINUS_INPLACE, backend, this, valueScalar);
        }
    }

    @Override
    public void subtract(@NotNull ITensor other) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        operationRecorder.recordOperation(OperationType.MINUS_INPLACE, backend, this, other);
    }

    // TODO: ACCELERATE WHERE OPERATIONS

    @Override
    public @NotNull ITensor where(byte condition, byte x, byte y) {
        ISciCoreBackend backend = getSciCoreBackend();
        ITensor result = backend.createTensor(DataType.INT8, getShape());
        for (int i = 0; i < numElements; i++) {
            result.setByByteFlat(getAsByteFlat(i) == condition ? x : y, i);
        }
        return result;
    }

    @Override
    public @NotNull ITensor where(short condition, short x, short y) {
        ISciCoreBackend backend = getSciCoreBackend();
        ITensor result = backend.createTensor(DataType.INT16, getShape());
        for (int i = 0; i < numElements; i++) {
            result.setByShortFlat(getAsShortFlat(i) == condition ? x : y, i);
        }
        return result;
    }

    @Override
    public @NotNull ITensor where(int condition, int x, int y) {
        ISciCoreBackend backend = getSciCoreBackend();
        ITensor result = backend.createTensor(DataType.INT32, getShape());
        for (int i = 0; i < numElements; i++) {
            result.setByIntFlat(getAsIntFlat(i) == condition ? x : y, i);
        }
        return result;
    }

    @Override
    public @NotNull ITensor where(long condition, long x, long y) {
        ISciCoreBackend backend = getSciCoreBackend();
        ITensor result = backend.createTensor(DataType.INT64, getShape());
        for (int i = 0; i < numElements; i++) {
            result.setByLongFlat(getAsLongFlat(i) == condition ? x : y, i);
        }
        return result;
    }

    @Override
    public @NotNull ITensor where(float condition, float x, float y) {
        ISciCoreBackend backend = getSciCoreBackend();
        ITensor result = backend.createTensor(DataType.FLOAT32, getShape());
        for (int i = 0; i < numElements; i++) {
            result.setByFloatFlat(getAsFloatFlat(i) == condition ? x : y, i);
        }
        return result;
    }

    @Override
    public @NotNull ITensor where(double condition, double x, double y) {
        ISciCoreBackend backend = getSciCoreBackend();
        ITensor result = backend.createTensor(DataType.FLOAT64, getShape());
        for (int i = 0; i < numElements; i++) {
            result.setByDoubleFlat(getAsDoubleFlat(i) == condition ? x : y, i);
        }
        return result;
    }

    @Override
    @NotNull
    public ITensor exp() {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.EXP, backend, this);
    }

    @Override
    @NotNull
    public ITensor softmax(int dimension) {
        try (ITensor exponentiated = exp()) {
            try (ITensor sum = exponentiated.reduceSum(dimension, true)) {
                return exponentiated.divide(sum);
            }
        }
    }

    @Override
    @NotNull
    public ITensor reduceSum(int dimension) {
        return reduceSum(dimension, false);
    }

    @Override
    @NotNull
    public ITensor reduceSum(int dimension, boolean keepDimensions) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor dimensionScalar = backend.createTensor(DataType.INT32, new long[0]);
             ITensor keepDimensionsScalar = backend.createTensor(DataType.BOOLEAN, new long[0])) {
            dimensionScalar.setIntFlat(dimension, 0);
            keepDimensionsScalar.setBooleanFlat(keepDimensions, 0);
            return operationRecorder.recordOperation(OperationType.REDUCE_SUM, backend, this, dimensionScalar, keepDimensionsScalar);
        }
    }

    @Override
    @NotNull
    public ITensor transpose() {
        long[] shape = getShape();
        return transpose(shape.length - 1, shape.length - 2);
    }

    @NotNull
    @Override
    public ITensor transpose(int dim1, int dim2) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor dim1Scalar = backend.createTensor(DataType.INT32, new long[0]);
             ITensor dim2Scalar = backend.createTensor(DataType.INT32, new long[0])) {
            dim1Scalar.setIntFlat(dim1, 0);
            dim2Scalar.setIntFlat(dim2, 0);
            return operationRecorder
                    .recordOperation(OperationType.TRANSPOSE,
                            OptionBundle.of(backend,
                                    Map.of(
                                            "dim1", dim1Scalar,
                                            "dim2", dim2Scalar
                                    )),
                            this);
        }
    }

    @Override
    public @NotNull ITensor contiguous() {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.CONTIGUOUS, backend, this);
    }

    @Override
    public @NotNull ITensor view(long @NotNull [] shape, long @NotNull [] strides) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor shapeTensor = backend.createTensor(DataType.INT64, new long[]{shape.length});
             ITensor stridesTensor = backend.createTensor(DataType.INT64, new long[]{strides.length})) {
            for (int i = 0; i < shape.length; i++) {
                shapeTensor.setLongFlat(shape[i], i);
            }
            for (int i = 0; i < strides.length; i++) {
                stridesTensor.setLongFlat(strides[i], i);
            }
            return operationRecorder.recordOperation(OperationType.RESHAPE, backend, this, shapeTensor, stridesTensor);
        }
    }

    @Override
    public @NotNull ITensor flatten(int dimension) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor startDimScalar = backend.createTensor(DataType.INT32, new long[0]);
             ITensor endDimScalar = backend.createTensor(DataType.INT32, new long[0])) {
            startDimScalar.setIntFlat(dimension, 0);
            endDimScalar.setIntFlat(dimension + 1, 0);
            return operationRecorder.recordOperation(OperationType.FLATTEN, OptionBundle.of(backend, Map.of("start_dim", startDimScalar, "end_dim", endDimScalar)), this);
        }
    }

    @Override
    public @NotNull ITensor concat(@NotNull ITensor tensor, int dimension) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor dimensionScalar = backend.createTensor(DataType.INT32, new long[0])) {
            dimensionScalar.setIntFlat(dimension, 0);
            return operationRecorder.recordOperation(OperationType.CONCAT, OptionBundle.of(backend, Map.of("dimension", dimensionScalar)), this, tensor);
        }
    }

    @Override
    public @NotNull ITensor pow(byte exponent) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor exponentScalar = backend.createTensor(DataType.INT8, new long[0])) {
            exponentScalar.setIntFlat(exponent, 0);
            return operationRecorder.recordOperation(OperationType.POW, backend, this, exponentScalar);
        }
    }

    @Override
    public @NotNull ITensor pow(short exponent) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor exponentScalar = backend.createTensor(DataType.INT16, new long[0])) {
            exponentScalar.setIntFlat(exponent, 0);
            return operationRecorder.recordOperation(OperationType.POW, backend, this, exponentScalar);
        }
    }

    @Override
    public @NotNull ITensor pow(int exponent) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor exponentScalar = backend.createTensor(DataType.INT32, new long[0])) {
            exponentScalar.setIntFlat(exponent, 0);
            return operationRecorder.recordOperation(OperationType.POW, backend, this, exponentScalar);
        }
    }

    @Override
    public @NotNull ITensor pow(long exponent) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor exponentScalar = backend.createTensor(DataType.INT64, new long[0])) {
            exponentScalar.setLongFlat(exponent, 0);
            return operationRecorder.recordOperation(OperationType.POW, backend, this, exponentScalar);
        }
    }

    @Override
    public @NotNull ITensor pow(float exponent) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor exponentScalar = backend.createTensor(DataType.FLOAT32, new long[0])) {
            exponentScalar.setFloatFlat(exponent, 0);
            return operationRecorder.recordOperation(OperationType.POW, backend, this, exponentScalar);
        }
    }

    @Override
    public @NotNull ITensor pow(double exponent) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor exponentScalar = backend.createTensor(DataType.FLOAT64, new long[0])) {
            exponentScalar.setDoubleFlat(exponent, 0);
            return operationRecorder.recordOperation(OperationType.POW, backend, this, exponentScalar);
        }
    }

    @Override
    public @NotNull ITensor pow(@NotNull ITensor exponent) {
        Validator.assertTrue(exponent.isScalar(), "Exponent must be scalar"); // TODO: Support non-scalar exponent
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.POW, backend, this, exponent);
    }

    @Override
    public @NotNull ITensor multiply(@NotNull ITensor other) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.MULTIPLY, backend, this, other);
    }

    @Override
    public @NotNull ITensor multiply(byte value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT8, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.MULTIPLY, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor multiply(short value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT16, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.MULTIPLY, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor multiply(int value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT32, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.MULTIPLY, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor multiply(long value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT64, new long[0])) {
            valueScalar.setLongFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.MULTIPLY, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor multiply(float value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.FLOAT32, new long[0])) {
            valueScalar.setFloatFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.MULTIPLY, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor multiply(double value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.FLOAT64, new long[0])) {
            valueScalar.setDoubleFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.MULTIPLY, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor relu() {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.RELU, backend, this);
    }

    @Override
    public @NotNull ITensor sigmoid() {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.SIGMOID, backend, this);
    }

    @Override
    public @NotNull ITensor tanh() {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.TANH, backend, this);
    }

    @Override
    public @NotNull ITensor log() {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.LOG, backend, this);
    }

    @Override
    public @NotNull ITensor argmax(int dimension) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor dimensionScalar = backend.createTensor(DataType.INT32, new long[0])) {
            dimensionScalar.setIntFlat(dimension, 0);
            return operationRecorder.recordOperation(OperationType.ARGMAX, backend, this, dimensionScalar);
        }
    }

    @Override
    public @NotNull ITensor lessThan(byte value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT8, new long[0])) {
            valueScalar.setByteFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.LESS_THAN, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor lessThan(short value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT16, new long[0])) {
            valueScalar.setShortFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.LESS_THAN, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor lessThan(int value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT32, new long[0])) {
            valueScalar.setIntFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.LESS_THAN, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor lessThan(long value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.INT64, new long[0])) {
            valueScalar.setLongFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.LESS_THAN, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor lessThan(float value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.FLOAT32, new long[0])) {
            valueScalar.setFloatFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.LESS_THAN, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor lessThan(double value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor valueScalar = backend.createTensor(DataType.FLOAT64, new long[0])) {
            valueScalar.setDoubleFlat(value, 0);
            return operationRecorder.recordOperation(OperationType.LESS_THAN, backend, this, valueScalar);
        }
    }

    @Override
    public @NotNull ITensor lessThan(@NotNull ITensor other) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.LESS_THAN, backend, this, other);
    }

    @Override
    public @NotNull ITensor compareElements(@NotNull ITensor other) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.COMPARE_ELEMENTS, backend, this, other);
    }

    @Override
    public @NotNull ITensor oneHot(long nClasses) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor nClassesScalar = backend.createTensor(DataType.INT64, new long[0])) {
            nClassesScalar.setLongFlat(nClasses, 0);
            return operationRecorder.recordOperation(OperationType.ONE_HOT, backend, this, nClassesScalar);
        }
    }

    @Override
    public @NotNull ITensor get(@NotNull ITensor... indicesTensors) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();

        ITensor[] inputs = new ITensor[indicesTensors.length + 1];
        inputs[0] = this;
        System.arraycopy(indicesTensors, 0, inputs, 1, indicesTensors.length);
        return operationRecorder.recordOperation(OperationType.GET, backend, inputs);
    }

    @Override
    public @NotNull ITensor to(@NotNull ISciCoreBackend backend) {
        ITensor newTensor = backend.createTensor(getDataType(), getShape());
        newTensor.setContents(this);
        return newTensor;
    }

    @Override
    public boolean isDisposed() {
        return disposed;
    }

    @Override
    public void dispose() {
        this.disposed = true;
    }

    protected void validateDataType(@NotNull DataType requestedDataType) {
        DataType ownDataType = getDataType();
        if (requestedDataType != ownDataType) {
            throw new IllegalArgumentException("Requested data type " + requestedDataType + " does not match data type of viewed tensor " + ownDataType);
        }
    }

    protected void validateIndices(long @NotNull [] indices) {
        long[] shape = getShape();
        if (indices.length > shape.length) {
            throw new IllegalArgumentException("Indices length is greater than tensor shape length");
        }
        for (int i = 0; i < indices.length; i++) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException("Index " + indices[i] + " is out of bounds for dimension " + i + " with shape " + shape[i]);
            }
        }
    }

    @Override
    public @NotNull ITensor cast(@NotNull DataType dataType) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        try (ITensor dataTypeScalar = backend.createTensor(DataType.INT32, new long[0])) {
            dataTypeScalar.setIntFlat(dataType.ordinal(), 0);
            return operationRecorder.recordOperation(OperationType.CAST, backend, this, dataTypeScalar);
        }
    }

    private boolean deReferenced = false;

    @Override
    public void close() {
        if (!deReferenced) {
            deReferenced = true;
        }
    }

    @Override
    public boolean isDeReferenced() {
        return deReferenced;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (!(obj instanceof ITensor other)) {
            return false;
        }

        long[] shape = getShape();
        if (!ShapeUtils.equals(shape, other.getShape())) {
            return false;
        }
        DataType dataType = getDataType();
        long[] index = new long[shape.length];
        if (dataType.isFloatingPoint()) {
            do {
                double v1 = getAsDouble(index);
                double v2 = other.getAsDouble(index);
                if (Math.abs(v1 - v2) > EPSILON + Math.ulp(v1) + Math.ulp(v2)) {
                    return false;
                }
            } while (ShapeUtils.incrementIndex(index, shape));
        } else {
            do {
                if (getAsLong(index) != other.getAsLong(index)) {
                    return false;
                }
            } while (ShapeUtils.incrementIndex(index, shape));
        }
        return true;
    }

    @Override
    public String toString() {
        long[] shape = getShape();
        StringBuilder sb = new StringBuilder(getClass().getSimpleName() + "(dtype=")
                .append(getDataType())
                .append(", ");
        long nElements = getNumberOfElements();
        boolean isNewLine = false;
        if (isScalar()) {
            sb.append("shape=")
                    .append(Arrays.toString(shape))
                    .append(", isScalar=true, data=");
            switch (getDataType()) {
                case INT8 -> sb.append(elementAsByte());
                case INT16 -> sb.append(elementAsShort());
                case INT32 -> sb.append(elementAsInt());
                case INT64 -> sb.append(elementAsLong());
                case FLOAT32 -> sb.append(formatFloat(elementAsFloat()));
                case FLOAT64 -> sb.append(formatFloat(elementAsDouble()));
                case BOOLEAN -> sb.append(elementAsBoolean());
            }
            sb.append(')');
            return sb.toString();
        } else {
            sb.append("shape=")
                    .append(Arrays.toString(shape))
                    .append(", data=");
            if (nElements >= 15) {
                sb.append('\n');
                isNewLine = true;
            }
        }
        sb.ensureCapacity(sb.length() + (int) nElements * 15);
        long[] index = new long[shape.length];
        boolean hasNext;
        long nElementsInDimension = 0;
        int nStartingDimensions;
        int nEndingDimensions;

        do {
            hasNext = false;
            nEndingDimensions = 0;

            // print tab, if new line
            if (isNewLine) {
                sb.append('\t');
            }

            // get starting dimensions
            {
                nStartingDimensions = 0;
                for (int dim = index.length - 1; dim >= 0; dim--) {
                    if (index[dim] == 0) {
                        nStartingDimensions++;
                    } else {
                        break;
                    }
                }
            }
            // print spaces to align with elements of previous line
            if (isNewLine) {
                sb.append(" ".repeat(Math.max(0, index.length - nStartingDimensions)));
            }
            // print starting brackets
            sb.append("[".repeat(Math.max(0, nStartingDimensions)));

            isNewLine = false;


            // print element
            switch (getDataType()) {
                case INT8 -> sb.append(getByte(index));
                case INT16 -> sb.append(getShort(index));
                case INT32 -> sb.append(getInt(index));
                case INT64 -> sb.append(getLong(index));
                case FLOAT32 -> sb.append(formatFloat(getFloat(index)));
                case FLOAT64 -> sb.append(formatFloat(getDouble(index)));
                case BOOLEAN -> sb.append(getBoolean(index));
            }
            nElementsInDimension++;

            boolean printDots = false;
            // increment index
            for (int dim = index.length - 1; dim >= 0; dim--) {
                if (index[dim] < shape[dim] - 1) {
                    index[dim]++;
                    // check if index hits 15 and has more than elements 15 to go to print '...'
                    if (index[dim] == 15 && shape[dim] > 15 * 2) {
                        index[dim] = shape[dim] - 15;
                        printDots = true;
                    }
                    hasNext = true;
                    break;
                }
                index[dim] = 0;
                nEndingDimensions++;
            }

            // ending dimensions
            sb.append("]".repeat(Math.max(0, nEndingDimensions)));

            if (hasNext) {
                sb.append(",");
                if (printDots) {
                    for (int i = 0; i < nEndingDimensions; i++) {
                        sb.append('\n');
                        isNewLine = true;
                    }
                    if (isNewLine) {
                        sb.append('\t');
                    }
                    sb.append(" ".repeat(Math.max(0, isNewLine ? (index.length - nStartingDimensions) : 1)));
                    sb.append("...,");
                    sb.append("\n".repeat(Math.max(0, nEndingDimensions - 1)));
                }
                if ((nElementsInDimension > 15 && nEndingDimensions > 0)) {
                    sb.append('\n');
                    isNewLine = true;
                }
                if (!isNewLine) {
                    sb.append(' ');
                }
            }

            if (nEndingDimensions > 0) {
                nElementsInDimension = 0;
            }

        } while (hasNext);
        sb.append(")");
        return sb.toString();
    }
}
