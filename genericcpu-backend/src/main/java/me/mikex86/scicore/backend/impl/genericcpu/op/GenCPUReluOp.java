package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.backend.impl.genericcpu.jni.ReluJNI;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.op.IDifferentiableUnaryOperation;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

public class GenCPUReluOp implements IDifferentiableUnaryOperation {

    @NotNull
    private final GenCPUBackend backend;

    public GenCPUReluOp(@NotNull GenCPUBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input) {
        long[] shape = input.getShape();
        long[] strides = input.getStrides();
        long nElements = ShapeUtils.getNumElements(shape);
        DataType dataType = input.getDataType();
        try (ITensor result = this.backend.createTensor(dataType, shape)) {
            DirectMemoryHandle inputHandle = input.getContentsAsDirectMemory();
            DirectMemoryHandle resultHandle = result.getContentsAsDirectMemory();
            ReluJNI.relu(inputHandle.getNativePtr(), resultHandle.getNativePtr(), nElements, dataType);
            return result.view(shape, strides);
        }
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input) {
        return new LazyTensor(backend, input.getShape(), input.getDataType());
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient input) {
        if (input.requiresGradients()) {
            ITensor inputTensor = input.getValue();
            long[] shape = inputTensor.getShape();
            long nElements = ShapeUtils.getNumElements(shape);
            DataType dataType = inputTensor.getDataType();
            try (ITensor gradient = this.backend.createTensor(dataType, shape)) {
                DirectMemoryHandle inputHandle = inputTensor.getContentsAsDirectMemory();
                DirectMemoryHandle gradientHandle = gradient.getContentsAsDirectMemory();
                ReluJNI.reluGradients(inputHandle.getNativePtr(), gradientHandle.getNativePtr(), nElements, dataType);

                ITensor finalGradient = gradient.multiply(upstreamGradient);
                input.accumulateGradient(finalGradient);
            }
        }
    }
}
