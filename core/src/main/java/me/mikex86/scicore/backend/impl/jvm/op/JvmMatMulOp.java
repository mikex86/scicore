package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.tensor.LazyTensor;
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
        return new LazyTensor(this.backend, resultShape, resultDataType);
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
        OptionBundle options = ctx.getOptionBundle();
        boolean transposeA = options.getOrDefault("transposeA", false);
        boolean transposeB = options.getOrDefault("transposeB", false);

        if (a.requiresGradients()) {
            ITensor dLdW;
            if (!transposeA) {
                // dL/dW = G @ X.T      # base case
                // if transposeB == true:
                //     dL/dW = G @ X.T.T    # the base case only applies when the transpose was APPLIED to the op input before the matrix multiplication
                //                          # During the forward pass a "virtual transpose" occurred, but this is not reflected in the graph.
                //                          # Thus, we need to transpose X again.
                //     dL/dW = G @ X        # X.T.T = X
                // else if transposeB == false:
                //     dL/dW = G @ X.T      # no virtual transpose occurred, because X here is what was actually used in the forward pass

                // interpretation: never transpose G, transpose X if transposeB == false
                dLdW = upstreamGradient.matmul(b.getValue(), false, !transposeB);
            } else {
                // Normally, if this were a transpose op node, this would compute the upstream gradients for
                // a transpose op, which would transpose it again as part of its gradient computation.
                // However, since we are merging a defacto transpose op into the matmul op, we would need to transpose
                // these gradients after dL/dW is computed. We avoid transposing by exploiting the identity:
                // B.T @ A.T = (A @ B).T

                // Derivation steps:
                // dL/dW_local = G @ X.T      # base case (fake local gradients)
                // if transposeB == true:
                //     dL/dW_local = G @ X.T.T    # virtual transpose occurred, so we need to transpose X again
                //     dL/dW_local = G @ X        # X.T.T = X
                //     dL/dW = (G @ X).T          # transpose because of would be transpose op chain rule
                //     dL/dW = X.T @ G.T          # apply identity
                // else if transposeB == false:
                //    dL/dW_local = G @ X.T       # no virtual transpose occurred, because X here is what was actually used in the forward pass
                //    dL/dW = (G @ X.T).T         # transpose because of would be transpose op chain rule
                //    dL/dW = X @ G.T             # apply identity

                // interpretation: always transpose G, transpose X if transposeB == true
                dLdW = b.getValue().matmul(upstreamGradient, transposeB, true);
            }
            a.accumulateGradient(dLdW);
        }

        if (b.requiresGradients()) {
            ITensor dLdX;
            if (!transposeB) {
                // dL/dX = W.T @ G      # base case
                // if transposeA == true:
                //     dL/dX = W.T.T @ G    # virtual transpose occurred, so we need to transpose W again
                //     dL/dX = W @ G        # W.T.T = W
                // else if transposeA == false:
                //     dL/dX = W.T @ G      # no virtual transpose occurred, because W here is what was actually used in the forward pass

                // interpretation: never transpose G, transpose W if transposeA == false
                dLdX = a.getValue().matmul(upstreamGradient, !transposeA, false);
            } else {
                // See above

                // Derivation steps:
                // dL/dX_local = W.T @ G    # base case (fake local gradients)
                // if transposeA == true:
                //     dL/dX_local = W.T.T @ G    # virtual transpose occurred, so we need to transpose W again
                //     dL/dX_local = W @ G        # W.T.T = W
                //     dL/dX = (W @ G).T          # transpose because of would be transpose op chain rule
                //     dL/dX = G.T @ W.T          # apply identity
                // else if transposeA == false:
                //    dL/dX_local = W.T @ G       # no virtual transpose occurred, because W here is what was actually used in the forward pass
                //    dL/dX = (W.T @ G).T         # transpose because of would be transpose op chain rule
                //    dL/dX = G.T @ W             # apply identity

                // interpretation: always transpose G, transpose W if transposeA == true
                dLdX = upstreamGradient.matmul(a.getValue(), true, transposeA);
            }
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
