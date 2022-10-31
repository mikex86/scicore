package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.op.IDifferentiableUnaryOperation;
import me.mikex86.scicore.graph.IGraph;
import org.jetbrains.annotations.NotNull;

public class JvmTransposeOp implements IDifferentiableUnaryOperation {

    @NotNull
    private final JvmBackend backend;

    public JvmTransposeOp(@NotNull JvmBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input) {
        long[] shape = input.getShape();
        if (shape.length > 2) {
            throw new IllegalArgumentException("transpose only supports 2D or lower tensors");
        }
        long[] resultShape = getResultShape(input);

        DataType dataType = input.getDataType();
        ITensor result = this.backend.createTensor(dataType, resultShape);

        long[] resultIndex = new long[resultShape.length];
        long[] inputIndex = new long[shape.length];

        // TODO: USE STRIDES SWAP
        if (dataType.isFloatingPoint()) {
            for (int i = 0; i < resultShape[0]; i++) {
                for (int j = 0; j < resultShape[1]; j++) {
                    resultIndex[0] = i;
                    resultIndex[1] = j;
                    inputIndex[0] = j;
                    inputIndex[1] = i;
                    double inputValue = input.getAsDouble(inputIndex);
                    result.setByDouble(inputValue, resultIndex);
                }
            }
        } else {
            for (int i = 0; i < resultShape[0]; i++) {
                for (int j = 0; j < resultShape[1]; j++) {
                    resultIndex[0] = i;
                    resultIndex[1] = j;
                    inputIndex[0] = j;
                    inputIndex[1] = i;

                    long inputValue = input.getAsLong(inputIndex);
                    result.setByLong(inputValue, resultIndex);
                }
            }
        }
        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input) {
        long[] shape = input.getShape();
        if (shape.length > 2) {
            throw new IllegalArgumentException("transpose only supports 2D or lower tensors");
        }
        long[] resultShape = getResultShape(input);

        DataType dataType = input.getDataType();
        return new LazyTensor(this.backend, resultShape, dataType);
    }

    private long @NotNull [] getResultShape(@NotNull ITensor input) {
        long[] shape = input.getShape();
        if (shape.length == 2) {
            return new long[]{shape[1], shape[0]};
        } else if (shape.length == 1) {
            return new long[]{1, shape[0]};
        } else {
            return new long[]{1, 1};
        }
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient input) {
        if (input.requiresGradients()) {
            input.accumulateGradient(upstreamGradient.transpose());
        }
    }
}
