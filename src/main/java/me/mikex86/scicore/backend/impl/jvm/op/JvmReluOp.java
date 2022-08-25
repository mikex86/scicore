package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.backend.impl.jvm.JvmTensor;
import me.mikex86.scicore.op.Graph;
import me.mikex86.scicore.op.IDifferentiableUnaryOperation;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

public class JvmReluOp implements IDifferentiableUnaryOperation {

    @NotNull
    private final JvmBackend backend;

    public JvmReluOp(@NotNull JvmBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input) {
        long[] shape = input.getShape();
        long nElements = ShapeUtils.getNumElements(shape);
        DataType dataType = input.getDataType();
        ITensor result = new JvmTensor(this.backend, dataType, shape);
        if (dataType.isFloatingPoint()) {
            for (long i = 0; i < nElements; i++) {
                double value = input.getAsDoubleFlat(i);
                result.setByDoubleFlat(Math.max(value, 0), i);
            }
        } else {
            for (long i = 0; i < nElements; i++) {
                long value = input.getAsLongFlat(i);
                result.setByLongFlat(Math.max(value, 0), i);
            }
        }
        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input) {
        return new LazyTensor(backend, input.getShape(), input.getDataType(), () -> perform(ctx, input));
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient input) {
        if (input.requiresGradients()) {
            ITensor inputTensor = input.getValue();
            long[] shape = inputTensor.getShape();
            long nElements = ShapeUtils.getNumElements(shape);
            DataType dataType = inputTensor.getDataType();
            ITensor gradient = new JvmTensor(this.backend, dataType, shape);
            if (dataType.isFloatingPoint()) {
                for (long i = 0; i < nElements; i++) {
                    double value = inputTensor.getAsDoubleFlat(i);
                    gradient.setByDoubleFlat(value > 0 ? 1 : 0, i);
                }
            } else {
                for (long i = 0; i < nElements; i++) {
                    long value = inputTensor.getAsLongFlat(i);
                    gradient.setByLongFlat(value > 0 ? 1 : 0, i);
                }
            }
            input.accumulateGradient(gradient.multiply(upstreamGradient));
        }
    }
}
