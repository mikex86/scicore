package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.graph.op.IDifferentiableSingleParametricOperation;
import me.mikex86.scicore.graph.op.IDifferentiableTrinaryOperation;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.tensor.View;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;
import java.util.Objects;

public class GenCpuReshapeOp implements IDifferentiableTrinaryOperation {

    @NotNull
    private final GenCPUBackend backend;

    public GenCpuReshapeOp(@NotNull GenCPUBackend backend) {
        this.backend = backend;
    }

    private ITensor getReshapedView(@NotNull ITensor tensor, long[] shape, long[] strides) {
        long shapeNumElements = ShapeUtils.getNumElements(shape);
        long tensorNumberOfElements = tensor.getNumberOfElements();
        if (shapeNumElements > tensorNumberOfElements) {
            throw new IllegalArgumentException("Cannot reshape tensor with " + tensorNumberOfElements + " elements to shape " + Arrays.toString(shape));
        }
        return new View(tensor, shape, 0, strides);
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor tensor, @NotNull ITensor shapeTensor, @NotNull ITensor stridesTensor) {
        if (shapeTensor.getShape().length != 1) {
            throw new IllegalArgumentException("shape must be a 1D tensor");
        }
        int numDims = Math.toIntExact(shapeTensor.getShape()[0]);
        long[] shape = new long[numDims];
        for (int i = 0; i < numDims; i++) {
            shape[i] = shapeTensor.getLong(i);
        }
        if (stridesTensor.getShape().length != 1) {
            throw new IllegalArgumentException("strides must be a 1D tensor");
        }
        if (stridesTensor.getShape()[0] != numDims) {
            throw new IllegalArgumentException("strides must have the same number of elements as shape");
        }
        long[] strides = new long[numDims];
        for (int i = 0; i < numDims; i++) {
            strides[i] = stridesTensor.getLong(i);
        }
        return getReshapedView(tensor, shape, strides);
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor inputTensor, @NotNull ITensor shapeTensor, @NotNull ITensor stridesTensor) {
        if (shapeTensor.getShape().length != 1) {
            throw new IllegalArgumentException("shape must be a 1D tensor");
        }
        int numDims = Math.toIntExact(shapeTensor.getShape()[0]);
        long[] shape = new long[numDims];
        for (int i = 0; i < numDims; i++) {
            shape[i] = shapeTensor.getLong(i);
        }
        return new LazyTensor(backend, shape, inputTensor.getDataType());
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient tensorNode, @NotNull IGraph.ITensorNodeWithGradient shapeNode, @NotNull IGraph.ITensorNodeWithGradient stridesTensorNode) {
        if (tensorNode.requiresGradients()) {
            long[] originalShape = tensorNode.getValue().getShape();
            long[] originalStrides = tensorNode.getValue().getStrides();
            tensorNode.accumulateGradient(getReshapedView(upstreamGradient, originalShape, originalStrides));
        }
    }
}
