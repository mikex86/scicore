package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.op.IDifferentiableUnaryOperation;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.util.Optional;

public class JvmSigmoidOp implements IDifferentiableUnaryOperation {

    @NotNull
    private final JvmBackend backend;

    public JvmSigmoidOp(@NotNull JvmBackend backend) {
        this.backend = backend;
    }

    private @NotNull ITensor sigmoid(@NotNull ITensor input) {
        long[] shape = input.getShape();
        long[] strides = input.getStrides();
        long nElements = ShapeUtils.getNumElements(shape);
        DataType dataType = input.getDataType();
        try (ITensor result = this.backend.createTensor(dataType, shape)) {
            if (dataType.isFloatingPoint()) {
                for (long i = 0; i < nElements; i++) {
                    double value = input.getAsDoubleFlat(i);
                    double sigmoid = 1.0 / (1 + Math.exp(-value));
                    result.setByDoubleFlat(sigmoid, i);
                }
            } else {
                for (long i = 0; i < nElements; i++) {
                    long value = input.getAsLongFlat(i);
                    long sigmoid = Math.round(1.0 / (1 + Math.exp(-value)));
                    result.setByLongFlat(sigmoid, i);
                }
            }
            return result.view(shape, strides);
        }
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input) {
        Optional<ITensor> sigmoid = ctx.getSavedTensor("sigmoid");
        if (sigmoid.isPresent()) {
            return sigmoid.get();
        } else {
            ITensor result = sigmoid(input);
            ctx.saveForBackward("sigmoid", result);
            return result;
        }
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input) {
        return new LazyTensor(backend, input.getShape(), input.getDataType());
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient input) {
        if (input.requiresGradients()) {
            ITensor sigmoid = ctx.getSavedTensorOrPopulateWith("sigmoid", () -> sigmoid(input.getValue()));
            try (ITensor negativeSigmoid = sigmoid.multiply(-1.0f)) {
                try (ITensor negativeSigmoidPlusOne = negativeSigmoid.plus(1.0f)) {
                    try (ITensor gradients = sigmoid.multiply(negativeSigmoidPlusOne)) {
                        input.accumulateGradient(gradients.multiply(upstreamGradient));
                    }
                }
            }
        }
    }
}
