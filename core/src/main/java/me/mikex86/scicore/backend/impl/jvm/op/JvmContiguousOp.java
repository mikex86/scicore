package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.op.IDifferentiableUnaryOperation;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.TensorContentUtils;
import org.jetbrains.annotations.NotNull;

public class JvmContiguousOp implements IDifferentiableUnaryOperation {

    @NotNull
    private final ISciCoreBackend backend;

    public JvmContiguousOp(@NotNull ISciCoreBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(Graph.@NotNull IOperationContext ctx, @NotNull ITensor input) {
        DirectMemoryHandle memoryHandle = TensorContentUtils.relayout(backend, input.getContentsAsDirectMemory(), input.getShape(), input.getStrides(), ShapeUtils.makeStrides(input.getShape()), input.getDataType());
        ITensor tensor = this.backend.createTensor(input.getDataType(), input.getShape());
        tensor.setContents(memoryHandle.asByteBuffer());
        return tensor;
    }

    @Override
    public @NotNull ITensor performLazily(Graph.@NotNull IOperationContext ctx, @NotNull ITensor input) {
        return new LazyTensor(backend, input.getShape(), input.getDataType());
    }

    @Override
    public void computeGradients(Graph.@NotNull IOperationContext ctx, @NotNull ITensor upstreamGradient, IGraph.@NotNull ITensorNodeWithGradient input) {
        if (input.requiresGradients()) {
            DirectMemoryHandle gradientMemory = TensorContentUtils.relayout(backend, upstreamGradient.getContentsAsDirectMemory(), upstreamGradient.getShape(), ShapeUtils.makeStrides(upstreamGradient.getShape()), upstreamGradient.getStrides(), upstreamGradient.getDataType());
            ITensor tensor = this.backend.createTensor(upstreamGradient.getDataType(), upstreamGradient.getShape());
            tensor.setContents(gradientMemory.asByteBuffer());
            input.accumulateGradient(tensor);
        }
    }
}
