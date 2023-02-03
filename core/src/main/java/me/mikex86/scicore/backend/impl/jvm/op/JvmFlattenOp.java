package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.OptionBundle;
import me.mikex86.scicore.graph.op.IDifferentiableUnaryOperation;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.tensor.View;
import org.jetbrains.annotations.NotNull;

public class JvmFlattenOp implements IDifferentiableUnaryOperation {

    @NotNull
    private final ISciCoreBackend backend;

    public JvmFlattenOp(@NotNull ISciCoreBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(Graph.@NotNull IOperationContext ctx, @NotNull ITensor input) {
        OptionBundle optionBundle = ctx.getOptionBundle();
        int dim = optionBundle.getInt("dimension").orElseThrow();
        long[] shape = input.getShape();
        long[] strides = input.getStrides();

        ITensor shapeTensor = backend.createTensor(DataType.INT64, new long[]{shape.length});
        for (int i = 0; i < shape.length; i++) {
            shapeTensor.setLong(shape[i], i);
        }
        ITensor stridesTensor = backend.createTensor(DataType.INT64, new long[]{strides.length});
        for (int i = 0; i < strides.length; i++) {
            stridesTensor.setLong(strides[i], i);
        }
        ctx.saveForBackward("shape", shapeTensor);
        ctx.saveForBackward("strides", stridesTensor);
        long[] newShape = new long[shape.length - dim];
        long[] newStrides = new long[strides.length - dim];
        long newStride = 1;
        for (int i = 0; i < newShape.length; i++) {
            newShape[i] = shape[i + dim];
            newStrides[i] = newStride;
            newStride *= newShape[i];
        }
        return new View(backend, input.getDataContainer(), newShape, 0, newStrides);
    }

    @Override
    public @NotNull ITensor performLazily(Graph.@NotNull IOperationContext ctx, @NotNull ITensor input) {
        OptionBundle optionBundle = ctx.getOptionBundle();
        int dim = optionBundle.getInt("dimension").orElseThrow();
        if (dim < 0 || dim >= input.getShape().length) {
            throw new IllegalArgumentException("Invalid dim: " + dim);
        }
        return new LazyTensor(backend, input.getShape(), input.getDataType());
    }

    @Override
    public void computeGradients(Graph.@NotNull IOperationContext ctx, @NotNull ITensor upstreamGradient, IGraph.@NotNull ITensorNodeWithGradient input) {
        if (input.requiresGradients()) {
            try (ITensor shapeTensor = ctx.getSavedTensor("shape").orElseThrow();
                 ITensor stridesTensor = ctx.getSavedTensor("strides").orElseThrow()) {
                long[] shape = new long[(int) shapeTensor.getShape()[0]];
                long[] strides = new long[(int) stridesTensor.getShape()[0]];
                for (int i = 0; i < shape.length; i++) {
                    shape[i] = shapeTensor.getLong(i);
                }
                for (int i = 0; i < strides.length; i++) {
                    strides[i] = stridesTensor.getLong(i);
                }
                ITensor gradient = new View(backend, upstreamGradient.getDataContainer(), shape, 0, strides);
                input.accumulateGradient(gradient);
            }
        }
    }
}
