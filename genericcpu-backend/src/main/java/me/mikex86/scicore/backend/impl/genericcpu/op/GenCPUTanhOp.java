package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.backend.impl.genericcpu.jni.TanhJNI;
import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
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

public class GenCPUTanhOp implements IDifferentiableUnaryOperation {

    @NotNull
    private final GenCPUBackend backend;

    public GenCPUTanhOp(@NotNull GenCPUBackend backend) {
        this.backend = backend;
    }

    private @NotNull ITensor tanh(@NotNull ITensor input) {
        long[] shape = input.getShape();
        long[] strides = input.getStrides();
        long nElements = ShapeUtils.getNumElements(shape);
        DataType dataType = input.getDataType();
        ITensor result = this.backend.createTensor(dataType, shape);
        DirectMemoryHandle inputMemoryHandle = input.getContentsAsDirectMemory();
        DirectMemoryHandle resultMemoryHandle = result.getContentsAsDirectMemory();
        TanhJNI.tanh(inputMemoryHandle.getNativePtr(), resultMemoryHandle.getNativePtr(), nElements, dataType);
        result = result.getReshapedView(shape, strides);
        return result;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input) {
        Optional<ITensor> tanh = ctx.getSavedTensor("tanh");
        if (tanh.isPresent()) {
            return tanh.get();
        } else {
            ITensor result = tanh(input);
            ctx.saveForBackward("tanh", result);
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
            ITensor tanh = ctx.getSavedTensorOrPopulateWith("tanh", () -> tanh(input.getValue()));
            try (ITensor tanhSquared = tanh.pow(2f)) {
                try (ITensor gradients = tanhSquared.leftMinus(1)) {
                    input.accumulateGradient(gradients.multiply(upstreamGradient));
                }
            }
        }
    }

}
