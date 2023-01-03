package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.backend.impl.genericcpu.jni.SigmoidJNI;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.op.IDifferentiableUnaryOperation;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.util.Optional;

public class GenCPUSigmoidOp implements IDifferentiableUnaryOperation {

    @NotNull
    private final GenCPUBackend backend;

    public GenCPUSigmoidOp(@NotNull GenCPUBackend backend) {
        this.backend = backend;
    }


    @NotNull
    private ITensor sigmoid(@NotNull ITensor x) {
        long[] shape = x.getShape();
        long[] strides = x.getStrides();
        long nElements = ShapeUtils.getNumElements(shape);
        DataType dataType = x.getDataType();
        ITensor result = backend.createTensor(dataType, shape);
        DirectMemoryHandle inputMemoryHandle = x.getContentsAsDirectMemory();
        DirectMemoryHandle resultMemoryHandle = result.getContentsAsDirectMemory();
        SigmoidJNI.sigmoid(inputMemoryHandle.getNativePtr(), resultMemoryHandle.getNativePtr(), nElements, dataType);
        result = result.view(shape, strides);
        return result;
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
            ITensor gradients = sigmoid.multiply(sigmoid.multiply(-1.0f).plus(1.0f));
            input.accumulateGradient(gradients.multiply(upstreamGradient));
        }
    }
}
