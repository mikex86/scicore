package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.OptionBundle;
import me.mikex86.scicore.graph.op.IUnaryOperation;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import org.jetbrains.annotations.NotNull;

public class GenCPUFillTriangleOp implements IUnaryOperation {

    @NotNull
    private final ISciCoreBackend backend;

    public GenCPUFillTriangleOp(@NotNull ISciCoreBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(Graph.@NotNull IOperationContext ctx, @NotNull ITensor input) {
        if (input.getShape().length < 2) {
            throw new IllegalArgumentException("Input must be at least 2-dimensional");
        }
        OptionBundle bundle = ctx.getOptionBundle();

        double topValue = bundle.getDouble("topValue").orElseThrow();
        double bottomValue = bundle.getDouble("bottomValue").orElseThrow();


        // TODO: OPTIMIZE
        long[] shape = input.getShape();
        ITensor oneBatchDim = input.view(-1, shape[shape.length - 2], shape[shape.length - 1]);
        long[] oneBatchShape = oneBatchDim.getShape();
        for (int i = 0; i < oneBatchShape[0]; i++) {
            for (int j = 0; j < oneBatchShape[1]; j++) {
                for (int k = 0; k < oneBatchShape[2]; k++) {
                    if (j < k) {
                        oneBatchDim.setByDouble(topValue, i, j, k);
                    } else {
                        oneBatchDim.setByDouble(bottomValue, i, j, k);
                    }
                }
            }
        }
        return oneBatchDim.view(shape);
    }

    @Override
    public @NotNull ITensor performLazily(Graph.@NotNull IOperationContext ctx, @NotNull ITensor input) {
        return new LazyTensor(backend, input.getShape(), input.getDataType());
    }
}
