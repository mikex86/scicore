package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.backend.impl.genericcpu.GenCPUTensor;
import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.op.Graph;
import me.mikex86.scicore.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

import static me.mikex86.scicore.backend.impl.genericcpu.jni.MatmulJNI.*;

public class GenCPUMatMulOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final GenCPUBackend backend;

    public GenCPUMatMulOp(@NotNull GenCPUBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shape = a.getShape();
        long[] otherShape = b.getShape();
        Validator.assertTrue(otherShape.length == 2, "Only 2D matrices are supported");
        Validator.assertTrue(shape.length == 2, "Only 2D matrices are supported");
        Validator.assertTrue(shape[1] == otherShape[0], "Shape mismatch. A.shape[1] != B.shape[0]");
        Validator.assertTrue(a.getDataType().isNumeric(), "Data type of A is not numeric");
        Validator.assertTrue(b.getDataType().isNumeric(), "Data type of B is not numeric");
        long[] resultShape = ShapeUtils.matrixMultiplyShape(shape, otherShape);
        DataType aDataType = a.getDataType();
        DataType bDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(aDataType, bDataType);

        GenCPUTensor result = new GenCPUTensor(this.backend, resultDataType, resultShape);

        int m = Math.toIntExact(shape[0]),
                n = Math.toIntExact(otherShape[1]),
                k = Math.toIntExact(shape[1]);

        DirectMemoryHandle aPtr = backend.getDirectMemoryManager().ensureDirect(a);
        DirectMemoryHandle bPtr = backend.getDirectMemoryManager().ensureDirect(b);

        matmul(OP_NONE, OP_NONE,
                    m, n, k,
                    aPtr.getNativePtr(),
                    getMatmulDataType(aDataType),
                    m,
                    bPtr.getNativePtr(),
                    getMatmulDataType(bDataType),
                    k,
                    result.getDataContainer().getMemoryHandle().getNativePtr(),
                    getMatmulDataType(resultDataType),
                    m
        );

        if (aPtr.canFree()) {
            aPtr.free();
        }
        if (bPtr.canFree()) {
            bPtr.free();
        }

        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shape = a.getShape();
        long[] otherShape = b.getShape();
        Validator.assertTrue(otherShape.length == 2, "Only 2D matrices are supported");
        Validator.assertTrue(shape.length == 2, "Only 2D matrices are supported");
        Validator.assertTrue(shape[1] == otherShape[0], "Shape mismatch. A.shape[1] != B.shape[0]");
        Validator.assertTrue(a.getDataType().isNumeric(), "Data type of A is not numeric");
        Validator.assertTrue(b.getDataType().isNumeric(), "Data type of B is not numeric");
        Validator.assertTrue(ShapeUtils.shapeFitsInInt(shape), "Shape of A is too large, no dimension must exceed Integer.MAX_VALUE");
        Validator.assertTrue(ShapeUtils.shapeFitsInInt(otherShape), "Shape of B is too large, no dimension must exceed Integer.MAX_VALUE");
        long[] resultShape = ShapeUtils.matrixMultiplyShape(shape, otherShape);
        DataType aDataType = a.getDataType();
        DataType bDataType = b.getDataType();
        // TODO: DATA TYPE VALIDATION
        DataType resultDataType = DataType.getLarger(aDataType, bDataType);
        return new LazyTensor(this.backend, resultShape, resultDataType, () -> perform(ctx, a, b));
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient a, @NotNull IGraph.ITensorNodeWithGradient b) {
        // See: https://cs231n.github.io/optimization-2/#mat (Stanford University CS231n: Deep Learning for Computer Vision)
        // Notation: WX = D

        // W = a, X = b, D = result
        // L = loss function (or more generally, the root of the graph that we derive in respect to)
        // G = upstream gradient = dL/dD

        // .T = transpose
        // @ = matrix multiplication

        // Gradients:
        // dL/dW = G @ X.T
        // dL/dX = W.T @ G

        // TODO: OPTIMIZE

        if (a.requiresGradients()) {
            ITensor dLdW = upstreamGradient.matmul(b.getValue().transpose());
            a.accumulateGradient(dLdW);
        }

        if (b.requiresGradients()) {
            ITensor dLdX = a.getValue().transpose().matmul(upstreamGradient);
            b.accumulateGradient(dLdX);
        }
    }
}
