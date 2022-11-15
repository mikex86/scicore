package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

public class JvmGetOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final JvmBackend backend;

    public JvmGetOp(@NotNull JvmBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input, @NotNull ITensor indices) {
        long[] inputShape = input.getShape();
        long[] indicesShape = indices.getShape();
        if (!indices.getDataType().isInteger()) {
            throw new IllegalArgumentException("Index tensor must be an integer type");
        }
        long[] indexIntoIndex = new long[indicesShape.length];
        long[] resultShape = new long[indicesShape.length + inputShape.length - 1];
        System.arraycopy(indicesShape, 0, resultShape, 0, indicesShape.length);
        System.arraycopy(inputShape, 1, resultShape, indicesShape.length, inputShape.length - 1);
        ITensor result = backend.createTensor(input.getDataType(), resultShape);
        do {
            long index = indices.getAsLong(indexIntoIndex);
            if (index < 0 || index >= inputShape[0]) {
                throw new IllegalArgumentException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(inputShape));
            }
            ITensor inputView = input.getView(index);
            result.setContents(indexIntoIndex, inputView);
        } while (ShapeUtils.incrementIndex(indexIntoIndex, indicesShape));
        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input, @NotNull ITensor indices) {
        long[] inputShape = input.getShape();
        long[] indicesShape = indices.getShape();
        if (!indices.getDataType().isInteger()) {
            throw new IllegalArgumentException("Index tensor must be an integer type");
        }
        long[] resultShape = new long[indicesShape.length + inputShape.length - 1];
        System.arraycopy(indicesShape, 0, resultShape, 0, indicesShape.length);
        System.arraycopy(inputShape, 1, resultShape, indicesShape.length, inputShape.length - 1);
        return new LazyTensor(backend, resultShape, input.getDataType());
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient a, @NotNull IGraph.ITensorNodeWithGradient b) {
        if (a.requiresGradients()) {
            ITensor input = a.getValue();
            ITensor indices = b.getValue();

            long[] inputShape = input.getShape();
            long[] indicesShape = indices.getShape();

            ITensor gradients = backend.createTensor(input.getDataType(), inputShape);

            long[] indexIntoIndex = new long[indicesShape.length];

            do {
                long index = indices.getAsLong(indexIntoIndex);
                ITensor upstreamGradientsView = upstreamGradient.getView(indexIntoIndex);
                input.getView(index).add(upstreamGradientsView);
            } while (ShapeUtils.incrementIndex(indexIntoIndex, indicesShape));

            a.accumulateGradient(gradients);
        }
    }
}
