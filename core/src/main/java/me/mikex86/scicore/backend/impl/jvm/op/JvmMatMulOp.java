package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.backend.impl.jvm.JvmTensor;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.op.Graph;
import me.mikex86.scicore.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.op.OptionBundle;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

public class JvmMatMulOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final JvmBackend backend;

    public JvmMatMulOp(@NotNull JvmBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shapeA = a.getShape();
        long[] shapeB = b.getShape();
        Validator.assertTrue(shapeB.length == 2, "Only 2D matrices are supported");
        Validator.assertTrue(shapeA.length == 2, "Only 2D matrices are supported");
        OptionBundle options = ctx.getOptionBundle();
        boolean transposeA = options.getOrDefault("transposeA", false);
        boolean transposeB = options.getOrDefault("transposeB", false);
        if (transposeA) {
            shapeA = new long[]{shapeA[1], shapeA[0]};
        }
        if (transposeB) {
            shapeB = new long[]{shapeB[1], shapeB[0]};
        }
        Validator.assertTrue(shapeA[1] == shapeB[0], "Shape mismatch. A.shape[1] != B.shape[0]");
        Validator.assertTrue(a.getDataType().isNumeric(), "Data type of A is not numeric");
        Validator.assertTrue(b.getDataType().isNumeric(), "Data type of B is not numeric");
        long[] resultShape = ShapeUtils.matrixMultiplyShape(shapeA, shapeB);
        DataType aDataType = a.getDataType();
        DataType bDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(aDataType, bDataType);

        ITensor result = this.backend.createTensor(resultDataType, resultShape);

        if (resultDataType.isFloatingPoint()) {
            for (int i = 0; i < resultShape[0]; i++) {
                for (int k = 0; k < shapeA[1]; k++) {
                    for (int j = 0; j < resultShape[1]; j++) {
                        double aValue = transposeA ? a.getAsDoubleFlat(k * resultShape[0] + i) : a.getAsDoubleFlat(i * shapeA[1] + k);
                        double bValue = transposeB ? b.getAsDoubleFlat(j * shapeB[0] + k) : b.getAsDoubleFlat(k * resultShape[1] + j);
                        double resultValue = result.getAsDoubleFlat(i * resultShape[1] + j);
                        result.setByDoubleFlat(resultValue + aValue * bValue, i * resultShape[1] + j);
                    }
                }
            }
        } else {
            for (int i = 0; i < resultShape[0]; i++) {
                for (int k = 0; k < shapeA[1]; k++) {
                    for (int j = 0; j < resultShape[1]; j++) {
                        long aValue = transposeA ? a.getAsLongFlat(k * resultShape[0] + i) : a.getAsLongFlat(i * shapeA[1] + k);
                        long bValue = transposeB ? b.getAsLongFlat(j * shapeB[0] + k) : b.getAsLongFlat(k * resultShape[1] + j);
                        long resultValue = result.getAsLongFlat(i * resultShape[1] + j);
                        result.setByLongFlat(resultValue + aValue * bValue, i * resultShape[1] + j);
                    }
                }
            }
        }

        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shapeA = a.getShape();
        long[] shapeB = b.getShape();
        Validator.assertTrue(shapeB.length == 2, "Only 2D matrices are supported");
        Validator.assertTrue(shapeA.length == 2, "Only 2D matrices are supported");
        OptionBundle options = ctx.getOptionBundle();
        boolean transposeA = options.getOrDefault("transposeA", false);
        boolean transposeB = options.getOrDefault("transposeB", false);
        if (transposeA) {
            shapeA = new long[]{shapeA[1], shapeA[0]};
        }
        if (transposeB) {
            shapeB = new long[]{shapeB[1], shapeB[0]};
        }
        Validator.assertTrue(shapeA[1] == shapeB[0], "Shape mismatch. A.shape[1] != B.shape[0]");
        Validator.assertTrue(a.getDataType().isNumeric(), "Data type of A is not numeric");
        Validator.assertTrue(b.getDataType().isNumeric(), "Data type of B is not numeric");
        long[] resultShape = ShapeUtils.matrixMultiplyShape(shapeA, shapeB);
        DataType ownDataType = a.getDataType();
        DataType otherDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(ownDataType, otherDataType);
        if (!resultDataType.isNumeric()) {
            throw new IllegalArgumentException("Cannot perform matrix multiplication on non-numeric data types");
        }
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
