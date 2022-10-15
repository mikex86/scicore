package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.backend.impl.genericcpu.jni.DivideJNI;
import me.mikex86.scicore.backend.impl.genericcpu.jni.MinusJNI;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.op.Graph;
import me.mikex86.scicore.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

public class GenCPUDivideOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final GenCPUBackend backend;

    public GenCPUDivideOp(@NotNull GenCPUBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shapeA = a.getShape();
        long[] shapeB = b.getShape();
        long[] finalShape = ShapeUtils.broadcastShapes(shapeA, shapeB);

        long[] stridesA = a.getStrides();
        long[] stridesB = b.getStrides();

        DataType ownDataType = a.getDataType();
        DataType otherDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(ownDataType, otherDataType);
        ITensor result = backend.createTensor(resultDataType, finalShape);
        long[] resultStrides = result.getStrides();

        DirectMemoryHandle aMemoryHandle = a.getContentsAsDirectMemory();
        DirectMemoryHandle bMemoryHandle = b.getContentsAsDirectMemory();
        DirectMemoryHandle resultMemoryHandle = result.getContentsAsDirectMemory();

        DivideJNI.divide(
                aMemoryHandle.getNativePtr(), shapeA, stridesA, a.getDataType(),
                bMemoryHandle.getNativePtr(), shapeB, stridesB, b.getDataType(),
                resultMemoryHandle.getNativePtr(), finalShape, resultStrides, result.getDataType());


        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        Validator.assertTrue(a.getDataType().isNumeric(), "A is not numeric");
        Validator.assertTrue(b.getDataType().isNumeric(), "B is not numeric");
        long[] shapeA = a.getShape();
        long[] shapeB = b.getShape();
        long[] finalShape = ShapeUtils.broadcastShapes(shapeA, shapeB);

        DataType ownDataType = a.getDataType();
        DataType otherDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(ownDataType, otherDataType);
        return new LazyTensor(backend, finalShape, resultDataType, () -> perform(ctx, a, b));
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradients, @NotNull IGraph.ITensorNodeWithGradient a, @NotNull IGraph.ITensorNodeWithGradient b) {
        // R = A / B
        // dL/dR = upstream gradients
        ITensor A = a.getValue();
        ITensor B = b.getValue();
        if (a.requiresGradients()) {
            // dL/dA = dL/dR * dR/dA
            // dR/dA = 1/dB
            // dL/dA = dL/dR * 1/B
            ITensor dRdA = backend.createTensor(DataType.getLarger(A.getDataType(), B.getDataType()), B.getShape());
            dRdA.fill(1);
            dRdA = dRdA.divide(B); // TODO: optimize this with an leftDivide operation so that you can do B.leftDiv(1) to express 1/B
            a.accumulateGradient(upstreamGradients.multiply(dRdA));
        }
        if (b.requiresGradients()) {
            // dL/dB = dL/dR * dR/dB
            // R = A * B^-1
            // dR/dB = -A * (B^-2)
            // dR/dB = -A * (1/B^2)
            // dL/dB = dL/dR * -A * (1/B^2)
            ITensor dRdB = backend.createTensor(B.getDataType(), B.getShape());
            dRdB.fill(1);
            dRdB = dRdB.divide(B.pow(2));
            dRdB = A.multiply(-1).multiply(dRdB);
            b.accumulateGradient(upstreamGradients.multiply(dRdB));
        }
    }
}