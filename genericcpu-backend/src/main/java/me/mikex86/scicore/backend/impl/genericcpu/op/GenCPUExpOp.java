package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.backend.impl.genericcpu.jni.ExpJNI;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.op.IDifferentiableUnaryOperation;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.util.Optional;

public class GenCPUExpOp implements IDifferentiableUnaryOperation {

    @NotNull
    private final GenCPUBackend backend;

    public GenCPUExpOp(@NotNull GenCPUBackend backend) {
        this.backend = backend;
    }

    @NotNull
    private ITensor exp(@NotNull ITensor x) {
        long[] shape = x.getShape();
        long nElements = ShapeUtils.getNumElements(shape);
        DataType dataType = x.getDataType();
        ITensor result = backend.createTensor(dataType, shape);
        DirectMemoryHandle inputMemoryHandle = x.getContentsAsDirectMemory();
        DirectMemoryHandle resultMemoryHandle = result.getContentsAsDirectMemory();
        ExpJNI.exp(inputMemoryHandle.getNativePtr(), resultMemoryHandle.getNativePtr(), nElements, dataType);
        return result;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input) {
        Optional<ITensor> expOpt = ctx.getSavedTensor("exp");
        if (expOpt.isPresent()) {
            return expOpt.get();
        } else {
            ITensor result = exp(input);
            ctx.saveForBackward("exp", result);
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
            ITensor exp = ctx.getSavedTensorOrPopulateWith("exp", () -> exp(input.getValue()));
            input.accumulateGradient(upstreamGradient.multiply(exp));
        }
    }
}
