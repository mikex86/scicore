package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.OptionBundle;
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
        long[] stridesA = a.getStrides();
        long[] stridesB = b.getStrides();
        Validator.assertTrue(shapeB.length == 2, "Only 2D matrices are supported");
        Validator.assertTrue(shapeA.length == 2, "Only 2D matrices are supported");
        assert stridesA.length == 2;
        assert stridesB.length == 2;
        OptionBundle options = ctx.getOptionBundle();
        boolean transposeA = options.getOrDefault("transposeA", false);
        boolean transposeB = options.getOrDefault("transposeB", false);
        if (!(stridesA[0] == shapeA[1] && stridesA[1] == 1)) {
            throw new IllegalArgumentException("Invalid strides for matrix A"); // TODO: Support strides
        }
        if (!(stridesB[0] == shapeB[1] && stridesB[1] == 1)) {
            throw new IllegalArgumentException("Invalid strides for matrix B"); // TODO: Support strides
        }

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

        long m = shapeA[0],
                n = shapeB[1],
                k = shapeA[1];

        long lda = stridesA[0];
        long ldb = stridesB[0];

        matmul(transposeA, transposeB,
                m, n, k,
                a, lda,
                b, ldb,
                result, n
        );

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

    private static void matmul(
            boolean transposeA,
            boolean transposeB,
            long m, long n, long k,
            ITensor a,
            long lda,
            ITensor b,
            long ldb,
            ITensor result,
            long ldc
    ) {
        if (result.getDataType().isFloatingPoint()) {
            for (long row = 0; row < m; row++) {
                for (long inner = 0; inner < k; inner++) {
                    for (long col = 0; col < n; col++) {
                        long aIdx = transposeA ?
                                inner * lda + row :
                                row * lda + inner;
                        long bIdx = transposeB ?
                                col * ldb + inner :
                                inner * ldb + col;
                        long cIdx = row * ldc + col;
                        double c = result.getAsDoubleFlat(cIdx);
                        result.setByDoubleFlat(c + a.getAsDoubleFlat(aIdx) * b.getAsDoubleFlat(bIdx), cIdx);
                    }
                }
            }
        } else {
            for (long row = 0; row < m; row++) {
                for (long inner = 0; inner < k; inner++) {
                    for (long col = 0; col < n; col++) {
                        long aIdx = transposeA ?
                                inner * lda + row :
                                row * lda + inner;
                        long bIdx = transposeB ?
                                col * ldb + inner :
                                inner * ldb + col;
                        long cIdx = row * ldc + col;
                        long c = result.getAsLongFlat(cIdx);
                        result.setByLongFlat(c + a.getAsLongFlat(aIdx) * b.getAsLongFlat(bIdx), cIdx);
                    }
                }
            }
        }
    }
}
