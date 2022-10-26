package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.backend.impl.genericcpu.jni.ArgmaxJNI;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.op.IDifferentiableSingleParametricOperation;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

public class GenCpuArgmaxOp implements IDifferentiableSingleParametricOperation<Integer> {

    @NotNull
    private final GenCPUBackend backend;

    public GenCpuArgmaxOp(@NotNull GenCPUBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor tensor, @Nullable Integer dimension) {
        Validator.assertTrue(dimension != null, "dimensionScalar must not be null");

        long[] shape = tensor.getShape();
        if (dimension == -1) {
            long[] outputShape = ShapeUtils.getReducedShape(shape, dimension, false);
            ITensor result = backend.createTensor(DataType.INT64, outputShape);
            ArgmaxJNI.argmax(
                    tensor.getContentsAsDirectMemory().getNativePtr(), tensor.getShape(), tensor.getStrides(),
                    result.getContentsAsDirectMemory().getNativePtr(), result.getShape(), result.getStrides(),
                    DataType.INT64, dimension
            );
            return result;
        }
        if (dimension < 0 || dimension >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds: " + dimension);
        }
        long[] reducedShape = new long[shape.length - 1];
        ShapeUtils.reduceShape(shape, reducedShape, dimension, false);

        ITensor result = backend.createTensor(DataType.INT64, reducedShape);
        ArgmaxJNI.argmax(
                tensor.getContentsAsDirectMemory().getNativePtr(), tensor.getShape(), tensor.getStrides(),
                result.getContentsAsDirectMemory().getNativePtr(), result.getShape(), result.getStrides(),
                DataType.INT64, dimension
        );
        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input, @Nullable Integer dimension) {
        Validator.assertTrue(dimension != null, "dimensionScalar must not be null");
        long[] reducedShape = ShapeUtils.getReducedShape(input.getShape(), dimension, false);
        return new LazyTensor(backend, reducedShape, DataType.INT64, () -> perform(ctx, input, dimension));
    }


    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.IDifferentiableNode node, @Nullable Integer integer) {
        // TODO: IMPLEMENT
        throw new UnsupportedOperationException("TODO: implement");
    }

    @Override
    public @NotNull Class<Integer> getType() {
        return Integer.class;
    }
}
