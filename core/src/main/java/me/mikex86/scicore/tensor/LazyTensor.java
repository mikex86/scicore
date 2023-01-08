package me.mikex86.scicore.tensor;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.GraphExecutor;
import me.mikex86.scicore.graph.IGraphRecorder;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.tensor.data.ITensorDataContainer;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.IOException;
import java.io.InputStream;
import java.nio.*;
import java.util.Arrays;
import java.util.Objects;

/**
 * Encapsulates tensor supplier. Lazily evaluates the supplier when the result is needed.
 */
public class LazyTensor extends AbstractTensor implements IDerivedTensor {

    private final long @NotNull [] resultShape;

    @NotNull
    private final DataType resultDataType;

    @Nullable
    private ITensor lazyResult;

    @NotNull
    private final ISciCoreBackend sciCoreBackend;

    public LazyTensor(@NotNull ISciCoreBackend backend, long @NotNull [] resultShape, @NotNull DataType resultDataType) {
        this.numElements = ShapeUtils.getNumElements(resultShape);
        this.resultShape = resultShape;
        this.resultDataType = resultDataType;
        this.sciCoreBackend = backend;
    }

    private LazyTensor(@NotNull ITensor tensor) {
        this.numElements = tensor.getNumberOfElements();
        this.resultShape = tensor.getShape();
        this.resultDataType = tensor.getDataType();
        this.lazyResult = tensor;
        this.sciCoreBackend = tensor.getSciCoreBackend();
    }

    @Override
    @NotNull
    public ITensor getView(long @NotNull ... indices) {
        long[] shape = getShape();
        validateIndices(indices);
        long[] strides = getStrides();

        long[] sliceShape = Arrays.copyOfRange(shape, indices.length, shape.length);
        long[] sliceStrides = ShapeUtils.makeStrides(sliceShape);

        long viewOffset = result() instanceof View view ? view.getOffset() : 0;

        long offset = viewOffset + ShapeUtils.getFlatIndex(indices, shape, strides);
        return new View(getSciCoreBackend(), getDataContainer(), sliceShape, offset, sliceStrides);
    }

    @NotNull
    @Override
    public ITensor result() {
        if (lazyResult == null) {
            Objects.requireNonNull(associatedGraphNode, "Lazy tensor must be associated with a graph node. This could mean the tensor was already disposed.");
            ISciCoreBackend backend = getSciCoreBackend();
            IGraphRecorder graphRecorder = backend.getOperationRecorder();
            try (Graph graph = graphRecorder.getExecutionGraphTo(backend, this)) {
                GraphExecutor graphExecutor = new GraphExecutor();
                graphExecutor.execute(graph);
            }
        }
        return lazyResult;
    }

    public void setResult(@Nullable ITensor result) {
        while (result instanceof IDerivedTensor lazyTensor) {
            result = lazyTensor.result();
        }
        this.lazyResult = result;
    }

    @NotNull
    public ITensor lazyCopy() {
        LazyTensor lazyTensor = new LazyTensor(sciCoreBackend, resultShape, resultDataType);
        lazyTensor.setResult(lazyResult);
        lazyTensor.setAssociatedGraphNode(associatedGraphNode);
        return lazyTensor;
    }

    public void forceReevaluation() {
        this.lazyResult = null;
    }

    public boolean hasResult() {
        return lazyResult != null;
    }

    /**
     * Wraps a tensor in a lazy tensor. Why would you want to do that?
     * Because you can append operations to the lazy tensor, and the
     * tensor will only be evaluated when the result is needed.
     * A normal tensor has no way of doing this.
     *
     * @param tensor the tensor to wrap
     * @return a lazy tensor
     */
    @NotNull
    public static LazyTensor wrap(@NotNull ITensor tensor) {
        return new LazyTensor(tensor);
    }

    @Override
    public @NotNull DataType getDataType() {
        return resultDataType;
    }

    @Override
    public long @NotNull [] getShape() {
        return resultShape;
    }

    @Override
    public long @NotNull [] getStrides() {
        return result().getStrides();
    }

    @Override
    public boolean getBooleanFlat(long flatIndex) {
        return result().getBooleanFlat(flatIndex);
    }

    @Override
    public void setBooleanFlat(boolean value, long flatIndex) {
        result().setBooleanFlat(value, flatIndex);
    }

    @Override
    public byte getByteFlat(long flatIndex) {
        return result().getByteFlat(flatIndex);
    }

    @Override
    public short getShortFlat(long flatIndex) {
        return result().getShortFlat(flatIndex);
    }

    @Override
    public int getIntFlat(long flatIndex) {
        return result().getIntFlat(flatIndex);
    }

    @Override
    public long getLongFlat(long flatIndex) {
        return result().getLongFlat(flatIndex);
    }

    @Override
    public float getFloatFlat(long flatIndex) {
        return result().getFloatFlat(flatIndex);
    }

    @Override
    public double getDoubleFlat(long flatIndex) {
        return result().getDoubleFlat(flatIndex);
    }

    @Override
    public void setByteFlat(byte value, long flatIndex) {
        result().setByteFlat(value, flatIndex);
    }

    @Override
    public void setShortFlat(short value, long flatIndex) {
        result().setShortFlat(value, flatIndex);
    }

    @Override
    public void setIntFlat(int value, long flatIndex) {
        result().setIntFlat(value, flatIndex);
    }

    @Override
    public void setLongFlat(long value, long flatIndex) {
        result().setLongFlat(value, flatIndex);
    }

    @Override
    public void setFloatFlat(float value, long flatIndex) {
        result().setFloatFlat(value, flatIndex);
    }

    @Override
    public void setDoubleFlat(double value, long flatIndex) {
        result().setDoubleFlat(value, flatIndex);
    }

    @Override
    public @NotNull ITensor copy() {
        return result().copy();
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull ITensor tensor) {
        result().setContentsWithOffset(startFlatIndex, tensor);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull ByteBuffer buffer) {
        result().setContentsWithOffset(startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull ShortBuffer buffer) {
        result().setContentsWithOffset(startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull IntBuffer buffer) {
        result().setContentsWithOffset(startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull LongBuffer buffer) {
        result().setContentsWithOffset(startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull FloatBuffer buffer) {
        result().setContentsWithOffset(startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, @NotNull DoubleBuffer buffer) {
        result().setContentsWithOffset(startFlatIndex, buffer);
    }

    @Override
    public void setContentsWithOffset(long startFlatIndex, boolean @NotNull [] buffer) {
        result().setContentsWithOffset(startFlatIndex, buffer);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, short value) {
        result().fillRegion(startFlatIndex, endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, byte value) {
        result().fillRegion(startFlatIndex, endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, int value) {
        result().fillRegion(startFlatIndex, endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, long value) {
        result().fillRegion(startFlatIndex, endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, float value) {
        result().fillRegion(startFlatIndex, endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, double value) {
        result().fillRegion(startFlatIndex, endFlatIndex, value);
    }

    @Override
    public void fillRegion(long startFlatIndex, long endFlatIndex, boolean value) {
        result().fillRegion(startFlatIndex, endFlatIndex, value);
    }

    @Override
    public @NotNull ISciCoreBackend getSciCoreBackend() {
        return this.sciCoreBackend;
    }

    @Override
    public @NotNull ITensorDataContainer getDataContainer() {
        return result().getDataContainer();
    }

    @Override
    public @NotNull DirectMemoryHandle getContentsAsDirectMemory() {
        return result().getContentsAsDirectMemory();
    }

    @Override
    public @NotNull DirectMemoryHandle getContentsAsDirectMemory(long startFlatIndex, long endFlatIndex) {
        return result().getContentsAsDirectMemory(startFlatIndex, endFlatIndex);
    }

    @Override
    public String toString() {
        if (hasResult()) {
            return "LazyTensor(" +
                   "result=" + result() +
                   ')';
        } else {
            return "LazyTensor(" +
                   "shape=" + ShapeUtils.toString(resultShape) +
                   ", dataType=" + resultDataType +
                   ", hasResult=false)";
        }
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof ITensor tensor)) {
            return false;
        }
        if (!Arrays.equals(tensor.getShape(), this.getShape())) {
            return false;
        }
        if (tensor.getDataType() != this.getDataType()) {
            return false;
        }
        if (obj == this) {
            return true;
        }
        return result().equals(obj);
    }

    @Override
    public <T extends ITensor> @Nullable T getIfIsType(@NotNull Class<T> typeClass) {
        return result().getIfIsType(typeClass);
    }

    @Override
    public void dispose() {
        super.dispose();
        if (hasResult()) {
            result().dispose();
        } else {
            associatedGraphNode = null;
        }
    }

    @Override
    public boolean isDisposed() {
        if (hasResult()) {
            return result().isDisposed();
        } else {
            return super.isDisposed();
        }
    }

    @Override
    public void readFrom(@NotNull InputStream inputStream) throws IOException {
        if (hasResult()) {
            result().readFrom(inputStream);
        } else {
            this.associatedGraphNode = null;
            this.lazyResult = this.sciCoreBackend.createTensor(resultDataType, resultShape);
            this.lazyResult.readFrom(inputStream);
        }
    }
}
