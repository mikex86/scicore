package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.backend.impl.genericcpu.GenCPUTensor;
import me.mikex86.scicore.backend.impl.genericcpu.GenCPUTensorDataContainer;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.op.IDifferentiableUnaryOperation;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.TensorContentUtils;
import org.jetbrains.annotations.NotNull;

public class GenCPUContiguousOp implements IDifferentiableUnaryOperation {

    @NotNull
    private final GenCPUBackend backend;

    public GenCPUContiguousOp(@NotNull GenCPUBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(Graph.@NotNull IOperationContext ctx, @NotNull ITensor input) {
        DirectMemoryHandle memoryHandle = TensorContentUtils.relayout(backend, input.getContentsAsDirectMemory(), input.getShape(), input.getStrides(), ShapeUtils.makeStrides(input.getShape()), input.getDataType());
        return new GenCPUTensor(backend, new GenCPUTensorDataContainer(backend.getDirectMemoryManager(), memoryHandle, input.getNumberOfElements(), input.getDataType()), input.getShape());
    }

    @Override
    public @NotNull ITensor performLazily(Graph.@NotNull IOperationContext ctx, @NotNull ITensor input) {
        return new LazyTensor(backend, input.getShape(), input.getDataType());
    }

    @Override
    public void computeGradients(Graph.@NotNull IOperationContext ctx, @NotNull ITensor upstreamGradient, IGraph.@NotNull ITensorNodeWithGradient input) {
        if (input.requiresGradients()) {
            DirectMemoryHandle gradientMemory = TensorContentUtils.relayout(backend, upstreamGradient.getContentsAsDirectMemory(), upstreamGradient.getShape(), ShapeUtils.makeStrides(upstreamGradient.getShape()), upstreamGradient.getStrides(), upstreamGradient.getDataType());
            ITensor tensor = new GenCPUTensor(backend, new GenCPUTensorDataContainer(backend.getDirectMemoryManager(), gradientMemory, upstreamGradient.getNumberOfElements(), upstreamGradient.getDataType()), upstreamGradient.getShape());
            input.accumulateGradient(tensor);
        }
    }
}
