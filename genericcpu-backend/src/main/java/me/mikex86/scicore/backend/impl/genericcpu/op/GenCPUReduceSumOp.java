package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.backend.impl.genericcpu.jni.ReduceSumJNI;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.op.IDifferentiableBiParametricOperation;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

public class GenCPUReduceSumOp implements IDifferentiableBiParametricOperation<Integer, Boolean> {

    @NotNull
    private final GenCPUBackend backend;

    public GenCPUReduceSumOp(@NotNull GenCPUBackend backend) {
        this.backend = backend;
    }

    @NotNull
    public GenCPUBackend getBackend() {
        return backend;
    }

    @NotNull
    @Override
    public ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor tensor, @Nullable Integer dimension, @Nullable Boolean keepDimensions) {
        Validator.validateNotNull(dimension, "Dimension must not be null");
        Validator.validateNotNull(keepDimensions, "KeepDimensions must not be null");

        DataType dataType = tensor.getDataType();
        long[] shape = tensor.getShape();
        if (dimension == -1) {
            long[] reducedShape = ShapeUtils.getReducedShape(shape, dimension, keepDimensions);
            ITensor result = backend.createTensor(dataType, reducedShape);
            DirectMemoryHandle aMemoryHandle = tensor.getContentsAsDirectMemory();
            DirectMemoryHandle resultMemoryHandle = result.getContentsAsDirectMemory();
            ReduceSumJNI.reduceSum(
                    aMemoryHandle.getNativePtr(), resultMemoryHandle.getNativePtr(),
                    dataType, shape, tensor.getStrides(),
                    reducedShape, result.getStrides(),
                    dimension, keepDimensions
            );
            return result;
        }
        if (dimension < 0 || dimension >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds: " + dimension);
        }
        long[] reducedShape = new long[shape.length - (keepDimensions ? 0 : 1)];
        ShapeUtils.reduceShape(shape, reducedShape, dimension, keepDimensions);

        ITensor result = backend.createTensor(dataType, reducedShape);
        DirectMemoryHandle aMemoryHandle = tensor.getContentsAsDirectMemory();
        DirectMemoryHandle resultMemoryHandle = result.getContentsAsDirectMemory();
        ReduceSumJNI.reduceSum(
                aMemoryHandle.getNativePtr(), resultMemoryHandle.getNativePtr(),
                dataType, shape, tensor.getStrides(),
                reducedShape, result.getStrides(),
                dimension, keepDimensions
        );
        return result;
    }

    @NotNull
    @Override
    public ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor tensor, @Nullable Integer dimension, @Nullable Boolean keepDimensions) {
        Validator.validateNotNull(dimension, "Dimension must not be null");
        Validator.validateNotNull(keepDimensions, "KeepDimensions must not be null");
        DataType dataType = tensor.getDataType();
        long[] shape = tensor.getShape();
        long[] outputShape = ShapeUtils.getReducedShape(shape, dimension, keepDimensions);
        return new LazyTensor(backend, outputShape, dataType);
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.IDifferentiableNode node, @Nullable Integer dimension, @Nullable Boolean keepDimensions) {
        Validator.validateNotNull(dimension, "Dimension must not be null");
        Validator.validateNotNull(keepDimensions, "KeepDimensions must not be null");

        if (node.requiresGradients()) {
            try (ITensor gradients = backend.createTensor(upstreamGradient.getDataType(), node.getValue().getShape())) {
                gradients.fill(1);
                ITensor finalGradients = gradients.multiply(upstreamGradient);

                node.accumulateGradient(finalGradients);
            }
        }
    }

    @Override
    public @NotNull Class<Integer> getFirstType() {
        return Integer.class;
    }

    @Override
    public @NotNull Class<Boolean> getSecondType() {
        return Boolean.class;
    }
}
