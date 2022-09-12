package me.mikex86.scicore;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.op.IGraphRecorder;
import me.mikex86.scicore.op.OperationType;
import me.mikex86.scicore.op.OptionBundle;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.Map;

import static me.mikex86.scicore.utils.StringUtils.formatFloat;

public abstract class AbstractTensor implements ITensor {

    // All implementations must set this variable!
    protected long numElements;

    @Override
    @NotNull
    public ITensor getView(long @NotNull ... indices) {
        long[] shape = getShape();
        validateIndices(indices);
        long[] strides = ShapeUtils.makeStrides(shape);

        long[] sliceShape = Arrays.copyOfRange(shape, indices.length, shape.length);
        long[] sliceStrides = Arrays.copyOfRange(strides, indices.length, strides.length);

        long offset = ShapeUtils.getFlatIndex(indices, strides);
        return new View(this, sliceShape, offset, sliceStrides);
    }

    @Override
    public long getNumberOfElements() {
        return this.numElements;
    }

    @Override
    public byte getByte(long @NotNull ... indices) {
        return getByteFlat(ShapeUtils.getFlatIndex(indices, getStrides()));
    }

    @Override
    public void setByte(byte value, long @NotNull ... indices) {
        setByteFlat(value, ShapeUtils.getFlatIndex(indices, getStrides()));
    }

    @Override
    public short getShort(long @NotNull ... indices) {
        return getShortFlat(ShapeUtils.getFlatIndex(indices, getStrides()));
    }

    @Override
    public void setShort(short value, long @NotNull ... indices) {
        setShortFlat(value, ShapeUtils.getFlatIndex(indices, getStrides()));
    }


    @Override
    public int getInt(long @NotNull ... indices) {
        return getIntFlat(ShapeUtils.getFlatIndex(indices, getStrides()));
    }

    @Override
    public void setInt(int value, long @NotNull ... indices) {
        setIntFlat(value, ShapeUtils.getFlatIndex(indices, getStrides()));
    }

    @Override
    public void setLong(long value, long @NotNull ... indices) {
        setLongFlat(value, ShapeUtils.getFlatIndex(indices, getStrides()));
    }

    @Override
    public long getLong(long @NotNull ... indices) {
        return getLongFlat(ShapeUtils.getFlatIndex(indices, getStrides()));
    }

    @Override
    public float getFloat(long @NotNull ... indices) {
        return getFloatFlat(ShapeUtils.getFlatIndex(indices, getStrides()));
    }

    @Override
    public void setFloat(float value, long @NotNull ... indices) {
        setFloatFlat(value, ShapeUtils.getFlatIndex(indices, getStrides()));
    }

    @Override
    public double getDouble(long @NotNull ... indices) {
        return getDoubleFlat(ShapeUtils.getFlatIndex(indices, getStrides()));
    }

    @Override
    public void setDouble(double value, long @NotNull ... indices) {
        setDoubleFlat(value, ShapeUtils.getFlatIndex(indices, getStrides()));
    }

    @Override
    public boolean getBoolean(long @NotNull ... indices) {
        return getBooleanFlat(ShapeUtils.getFlatIndex(indices, getStrides()));
    }

    @Override
    public void setBoolean(boolean value, long @NotNull ... indices) {
        setBooleanFlat(value, ShapeUtils.getFlatIndex(indices, getStrides()));
    }

    @Override
    @NotNull
    public ITensor matmul(@NotNull ITensor other) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.MATMUL, this, other);
    }

    @Override
    public @NotNull ITensor matmul(@NotNull ITensor other, boolean transposeSelf, boolean transposeOther) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor transposeSelfTensor, transposeOtherTensor;
        {
            transposeSelfTensor = backend.createTensor(DataType.BOOLEAN, new long[]{1});
            transposeSelfTensor.setBooleanFlat(transposeSelf, 0);
        }
        {
            transposeOtherTensor = backend.createTensor(DataType.BOOLEAN, new long[]{1});
            transposeOtherTensor.setBooleanFlat(transposeOther, 0);
        }
        return operationRecorder.recordOperation(OperationType.MATMUL, OptionBundle.of(
                Map.of(
                        "transposeA", transposeSelfTensor,
                        "transposeB", transposeOtherTensor
                )
        ), this, other);
    }

    @Override
    @NotNull
    public ITensor divided(@NotNull ITensor other) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.DIVIDED, this, other);
    }

    @Override
    public @NotNull ITensor divided(byte value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor valueScalar;
        {
            valueScalar = backend.createTensor(DataType.INT8, new long[]{1});
            valueScalar.setIntFlat(value, 0);
        }
        return operationRecorder.recordOperation(OperationType.DIVIDED, this, valueScalar);
    }

    @Override
    public @NotNull ITensor divided(short value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor valueScalar;
        {
            valueScalar = backend.createTensor(DataType.INT16, new long[]{1});
            valueScalar.setIntFlat(value, 0);
        }
        return operationRecorder.recordOperation(OperationType.DIVIDED, this, valueScalar);
    }

    @Override
    public @NotNull ITensor divided(int value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor valueScalar;
        {
            valueScalar = backend.createTensor(DataType.INT32, new long[]{1});
            valueScalar.setIntFlat(value, 0);
        }
        return operationRecorder.recordOperation(OperationType.DIVIDED, this, valueScalar);
    }

    @Override
    public @NotNull ITensor divided(long value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor valueScalar;
        {
            valueScalar = backend.createTensor(DataType.INT64, new long[]{1});
            valueScalar.setLongFlat(value, 0);
        }
        return operationRecorder.recordOperation(OperationType.DIVIDED, this, valueScalar);
    }

    @Override
    public @NotNull ITensor divided(float value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor valueScalar;
        {
            valueScalar = backend.createTensor(DataType.FLOAT32, new long[]{1});
            valueScalar.setFloatFlat(value, 0);
        }
        return operationRecorder.recordOperation(OperationType.DIVIDED, this, valueScalar);
    }

    @Override
    public @NotNull ITensor divided(double value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor valueScalar;
        {
            valueScalar = backend.createTensor(DataType.FLOAT64, new long[]{1});
            valueScalar.setDoubleFlat(value, 0);
        }
        return operationRecorder.recordOperation(OperationType.DIVIDED, this, valueScalar);
    }

    @Override
    public @NotNull ITensor plus(byte value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor valueScalar;
        {
            valueScalar = backend.createTensor(DataType.INT8, new long[]{1});
            valueScalar.setIntFlat(value, 0);
        }
        return operationRecorder.recordOperation(OperationType.PLUS, this, valueScalar);
    }

    @Override
    public @NotNull ITensor plus(short value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor valueScalar;
        {
            valueScalar = backend.createTensor(DataType.INT16, new long[]{1});
            valueScalar.setIntFlat(value, 0);
        }
        return operationRecorder.recordOperation(OperationType.PLUS, this, valueScalar);
    }

    @Override
    public @NotNull ITensor plus(int value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor valueScalar;
        {
            valueScalar = backend.createTensor(DataType.INT32, new long[]{1});
            valueScalar.setIntFlat(value, 0);
        }
        return operationRecorder.recordOperation(OperationType.PLUS, this, valueScalar);
    }

    @Override
    public @NotNull ITensor plus(long value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor valueScalar;
        {
            valueScalar = backend.createTensor(DataType.INT64, new long[]{1});
            valueScalar.setLongFlat(value, 0);
        }
        return operationRecorder.recordOperation(OperationType.PLUS, this, valueScalar);
    }

    @Override
    public @NotNull ITensor plus(float value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor valueScalar;
        {
            valueScalar = backend.createTensor(DataType.FLOAT32, new long[]{1});
            valueScalar.setFloatFlat(value, 0);
        }
        return operationRecorder.recordOperation(OperationType.PLUS, this, valueScalar);
    }

    @Override
    public @NotNull ITensor plus(double value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor valueScalar;
        {
            valueScalar = backend.createTensor(DataType.FLOAT64, new long[]{1});
            valueScalar.setDoubleFlat(value, 0);
        }
        return operationRecorder.recordOperation(OperationType.PLUS, this, valueScalar);
    }

    @Override
    @NotNull
    public ITensor plus(@NotNull ITensor other) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.PLUS, this, other);
    }


    @Override
    public @NotNull ITensor minus(byte value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor valueScalar;
        {
            valueScalar = backend.createTensor(DataType.INT8, new long[]{1});
            valueScalar.setIntFlat(value, 0);
        }
        return operationRecorder.recordOperation(OperationType.MINUS, this, valueScalar);
    }

    @Override
    public @NotNull ITensor minus(short value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor valueScalar;
        {
            valueScalar = backend.createTensor(DataType.INT16, new long[]{1});
            valueScalar.setIntFlat(value, 0);
        }
        return operationRecorder.recordOperation(OperationType.MINUS, this, valueScalar);
    }

    @Override
    public @NotNull ITensor minus(int value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor valueScalar;
        {
            valueScalar = backend.createTensor(DataType.INT32, new long[]{1});
            valueScalar.setIntFlat(value, 0);
        }
        return operationRecorder.recordOperation(OperationType.MINUS, this, valueScalar);
    }

    @Override
    public @NotNull ITensor minus(long value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor valueScalar;
        {
            valueScalar = backend.createTensor(DataType.INT64, new long[]{1});
            valueScalar.setLongFlat(value, 0);
        }
        return operationRecorder.recordOperation(OperationType.MINUS, this, valueScalar);
    }

    @Override
    public @NotNull ITensor minus(float value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor valueScalar;
        {
            valueScalar = backend.createTensor(DataType.FLOAT32, new long[]{1});
            valueScalar.setFloatFlat(value, 0);
        }
        return operationRecorder.recordOperation(OperationType.MINUS, this, valueScalar);
    }

    @Override
    public @NotNull ITensor minus(double value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor valueScalar;
        {
            valueScalar = backend.createTensor(DataType.FLOAT64, new long[]{1});
            valueScalar.setDoubleFlat(value, 0);
        }
        return operationRecorder.recordOperation(OperationType.MINUS, this, valueScalar);
    }

    @Override
    @NotNull
    public ITensor minus(@NotNull ITensor other) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.MINUS, this, other);
    }

    @Override
    @NotNull
    public ITensor exp() {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.EXP, this);
    }

    @Override
    @NotNull
    public ITensor softmax(int dimension) {
        ITensor exponentiated = exp();
        ITensor sum = exponentiated.reduceSum(dimension, true);
        return exponentiated.divided(sum);
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
        ITensor dimensionScalar;
        {
            dimensionScalar = backend.createTensor(DataType.INT32, new long[]{1});
            dimensionScalar.setIntFlat(dimension, 0);
        }
        ITensor keepDimensionsScalar;
        {
            keepDimensionsScalar = backend.createTensor(DataType.BOOLEAN, new long[]{1});
            keepDimensionsScalar.setBooleanFlat(keepDimensions, 0);
        }
        return operationRecorder.recordOperation(OperationType.REDUCE_SUM, this, dimensionScalar, keepDimensionsScalar);
    }

    @Override
    @NotNull
    public ITensor transpose() {
        // TODO: OPTIMIZE TRANSPOSE AS TRANSPOSED VIEW
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.TRANSPOSE, this);
    }

    @Override
    public @NotNull ITensor pow(byte exponent) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor exponentScalar;
        {
            exponentScalar = backend.createTensor(DataType.INT8, new long[]{1});
            exponentScalar.setIntFlat(exponent, 0);
        }
        return operationRecorder.recordOperation(OperationType.POW, this, exponentScalar);
    }

    @Override
    public @NotNull ITensor pow(short exponent) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor exponentScalar;
        {
            exponentScalar = backend.createTensor(DataType.INT16, new long[]{1});
            exponentScalar.setIntFlat(exponent, 0);
        }
        return operationRecorder.recordOperation(OperationType.POW, this, exponentScalar);
    }

    @Override
    public @NotNull ITensor pow(int exponent) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor exponentScalar;
        {
            exponentScalar = backend.createTensor(DataType.INT32, new long[]{1});
            exponentScalar.setIntFlat(exponent, 0);
        }
        return operationRecorder.recordOperation(OperationType.POW, this, exponentScalar);
    }

    @Override
    public @NotNull ITensor pow(long exponent) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor exponentScalar;
        {
            exponentScalar = backend.createTensor(DataType.INT64, new long[]{1});
            exponentScalar.setLongFlat(exponent, 0);
        }
        return operationRecorder.recordOperation(OperationType.POW, this, exponentScalar);
    }

    @Override
    public @NotNull ITensor pow(float exponent) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor exponentScalar;
        {
            exponentScalar = backend.createTensor(DataType.FLOAT32, new long[]{1});
            exponentScalar.setFloatFlat(exponent, 0);
        }
        return operationRecorder.recordOperation(OperationType.POW, this, exponentScalar);
    }

    @Override
    public @NotNull ITensor pow(double exponent) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor exponentScalar;
        {
            exponentScalar = backend.createTensor(DataType.FLOAT64, new long[]{1});
            exponentScalar.setDoubleFlat(exponent, 0);
        }
        return operationRecorder.recordOperation(OperationType.POW, this, exponentScalar);
    }

    @Override
    public @NotNull ITensor pow(@NotNull ITensor exponent) {
        Validator.assertTrue(exponent.isScalar(), "Exponent must be scalar"); // TODO: Support non-scalar exponent
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.POW, this, exponent);
    }

    @Override
    public @NotNull ITensor multiply(@NotNull ITensor other) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.MULTIPLY, this, other);
    }

    @Override
    public @NotNull ITensor multiply(byte value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor valueScalar;
        {
            valueScalar = backend.createTensor(DataType.INT8, new long[]{1});
            valueScalar.setIntFlat(value, 0);
        }
        return operationRecorder.recordOperation(OperationType.MULTIPLY, this, valueScalar);
    }

    @Override
    public @NotNull ITensor multiply(short value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor valueScalar;
        {
            valueScalar = backend.createTensor(DataType.INT16, new long[]{1});
            valueScalar.setIntFlat(value, 0);
        }
        return operationRecorder.recordOperation(OperationType.MULTIPLY, this, valueScalar);
    }

    @Override
    public @NotNull ITensor multiply(int value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor valueScalar;
        {
            valueScalar = backend.createTensor(DataType.INT32, new long[]{1});
            valueScalar.setIntFlat(value, 0);
        }
        return operationRecorder.recordOperation(OperationType.MULTIPLY, this, valueScalar);
    }

    @Override
    public @NotNull ITensor multiply(long value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor valueScalar;
        {
            valueScalar = backend.createTensor(DataType.INT64, new long[]{1});
            valueScalar.setLongFlat(value, 0);
        }
        return operationRecorder.recordOperation(OperationType.MULTIPLY, this, valueScalar);
    }

    @Override
    public @NotNull ITensor multiply(float value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor valueScalar;
        {
            valueScalar = backend.createTensor(DataType.FLOAT32, new long[]{1});
            valueScalar.setFloatFlat(value, 0);
        }
        return operationRecorder.recordOperation(OperationType.MULTIPLY, this, valueScalar);
    }

    @Override
    public @NotNull ITensor multiply(double value) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor valueScalar;
        {
            valueScalar = backend.createTensor(DataType.FLOAT64, new long[]{1});
            valueScalar.setDoubleFlat(value, 0);
        }
        return operationRecorder.recordOperation(OperationType.MULTIPLY, this, valueScalar);
    }

    @Override
    public @NotNull ITensor relu() {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.RELU, this);
    }

    @Override
    public @NotNull ITensor sigmoid() {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.SIGMOID, this);
    }

    @Override
    public @NotNull ITensor argmax(int dimension) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        ITensor dimensionScalar;
        {
            dimensionScalar = backend.createTensor(DataType.INT32, new long[]{1});
            dimensionScalar.setIntFlat(dimension, 0);
        }
        return operationRecorder.recordOperation(OperationType.ARGMAX, this, dimensionScalar);
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
        ITensor dataTypeScalar;
        {
            dataTypeScalar = backend.createTensor(DataType.INT32, new long[]{1});
            dataTypeScalar.setIntFlat(dataType.ordinal(), 0);
        }
        return operationRecorder.recordOperation(OperationType.CAST, this, dataTypeScalar);
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
        long nElements = ShapeUtils.getNumElements(shape);
        DataType dataType = getDataType();
        if (dataType != other.getDataType()) {
            return false;
        }
        if (dataType.isFloatingPoint()) {
            for (long i = 0; i < nElements; i++) {
                double a = getAsDoubleFlat(i);
                double b = other.getAsDoubleFlat(i);
                if (Math.abs(a - b) > EPSILON) {
                    return false;
                }
            }
        } else {
            for (long i = 0; i < nElements; i++) {
                long a = getAsLongFlat(i);
                long b = other.getAsLongFlat(i);
                if (a != b) {
                    return false;
                }
            }
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
        sb.ensureCapacity(sb.length() + (int) nElements * 10);
        ITensorIterator iterator = iterator();
        int nElementsInLine = 0;
        while (iterator.hasNext()) {
            long nStartingDimensions = iterator.getNumStartingDimensions();
            long nEndingDimensions = iterator.getNumEndingDimensions();
            if (isNewLine) {
                sb.append("\t");
            }
            if (isNewLine) {
                for (int i = 0; i < shape.length - nStartingDimensions; i++) {
                    sb.append(" ");
                }
            }
            for (long i = 0; i < nStartingDimensions; i++) {
                sb.append("[");
            }
            switch (iterator.getDataType()) {
                case INT8 -> sb.append(iterator.getByte());
                case INT16 -> sb.append(iterator.getShort());
                case INT32 -> sb.append(iterator.getInt());
                case INT64 -> sb.append(iterator.getLong());
                case FLOAT32 -> sb.append(formatFloat(iterator.getFloat()));
                case FLOAT64 -> sb.append(formatFloat(iterator.getDouble()));
                case BOOLEAN -> sb.append(iterator.getBoolean());
            }
            for (long i = 0; i < nEndingDimensions; i++) {
                sb.append("]");
            }
            nElementsInLine++;
            iterator.moveNext();
            if (!iterator.hasNext()) {
                continue;
            }
            sb.append(",");
            if (nEndingDimensions > 0 && nElementsInLine >= 15) {
                sb.append("\n");
                isNewLine = true;
                nElementsInLine = 0;
            } else {
                sb.append(" ");
                isNewLine = false;
            }
        }
        sb.append(")");
        return sb.toString();
    }
}
