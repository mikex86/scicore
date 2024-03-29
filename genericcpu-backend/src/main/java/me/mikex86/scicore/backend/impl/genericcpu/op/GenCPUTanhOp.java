package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.backend.impl.genericcpu.jni.TanhJNI;
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
        try (ITensor result = this.backend.createTensor(dataType, shape)) {
            DirectMemoryHandle inputMemoryHandle = input.getContentsAsDirectMemory();
            DirectMemoryHandle resultMemoryHandle = result.getContentsAsDirectMemory();
            TanhJNI.tanh(inputMemoryHandle.getNativePtr(), resultMemoryHandle.getNativePtr(), nElements, dataType);
            return result.view(shape, strides);
        }
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
            ITensor inputValue = input.getValue();
            DataType dataType = inputValue.getDataType();
            long[] shape = inputValue.getShape();
            long nElements = ShapeUtils.getNumElements(shape);
            ITensor tanh = ctx.getSavedTensorOrPopulateWith("tanh", () -> tanh(inputValue));
            try (ITensor gradient = this.backend.createTensor(dataType, shape)) {
                DirectMemoryHandle savedTanhHandle = tanh.getContentsAsDirectMemory();
                DirectMemoryHandle gradientHandle = gradient.getContentsAsDirectMemory();
                TanhJNI.tanhGradients(savedTanhHandle.getNativePtr(), gradientHandle.getNativePtr(), nElements, dataType);
                ITensor finalGradient = gradient.multiply(upstreamGradient);
                input.accumulateGradient(finalGradient);
            }
        }
    }

}
