package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.OptionBundle;
import me.mikex86.scicore.utils.GradientUtil;
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

        Validator.assertTrue(shapeA.length == 2 || shapeA.length == 3, "Only 2D and 3D tensors are supported");
        Validator.assertTrue(shapeB.length == 2 || shapeB.length == 3, "Only 2D and 3D tensors are supported");

        OptionBundle options = ctx.getOptionBundle();
        boolean transposeA = options.getOrDefault("transposeA", false);
        boolean transposeB = options.getOrDefault("transposeB", false);

        // TODO: Support more complex strides
        if (stridesA[stridesA.length - 1] != 1 &&
            stridesA[stridesA.length - 1] == shapeA[shapeA.length - 1]
            && stridesA[stridesA.length - 2] == 1) {
            transposeA = !transposeA;
        }

        if (stridesB[stridesB.length - 1] != 1 &&
            stridesB[stridesB.length - 1] == shapeB[shapeB.length - 1]
            && stridesB[stridesB.length - 2] == 1) {
            transposeB = !transposeB;
        }

        if (transposeA) {
            if (shapeA.length == 2) {
                shapeA = new long[]{shapeA[shapeA.length - 1], shapeA[shapeA.length - 2]};
            } else {
                shapeA = new long[]{shapeA[0], shapeA[shapeA.length - 1], shapeA[shapeA.length - 2]};
            }
        }
        if (transposeB) {
            if (shapeB.length == 2) {
                shapeB = new long[]{shapeB[shapeB.length - 1], shapeB[shapeB.length - 2]};
            } else {
                shapeB = new long[]{shapeB[0], shapeB[shapeB.length - 1], shapeB[shapeB.length - 2]};
            }
        }
        Validator.assertTrue(shapeA[shapeA.length - 1] == shapeB[shapeB.length - 2], "Matrix multiplication shape mismatch: " + ShapeUtils.toString(shapeA) + " x " + ShapeUtils.toString(shapeB));
        Validator.assertTrue(a.getDataType().isNumeric(), "Data type of A is not numeric");
        Validator.assertTrue(b.getDataType().isNumeric(), "Data type of B is not numeric");

        if (shapeA.length == 3 && shapeB.length == 3) {
            Validator.assertTrue(shapeA[0] == shapeB[0] || shapeA[0] == 1 || shapeB[0] == 1, "Batch size mismatch");
        }

        long[] resultShape = ShapeUtils.matrixMultiplyShape(shapeA, shapeB);
        DataType aDataType = a.getDataType();
        DataType bDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(aDataType, bDataType);

        ITensor result = this.backend.createTensor(resultDataType, resultShape);

        long[] resultStrides = result.getStrides();

        int m = Math.toIntExact(shapeA[shapeA.length - 2]),
                n = Math.toIntExact(shapeB[shapeB.length - 1]),
                k = Math.toIntExact(shapeA[shapeA.length - 1]);

        int lda = transposeA ? m : k;
        int ldb = transposeB ? k : n;

        long aBatchSize = shapeA.length == 3 ? shapeA[0] : 1;
        long bBatchSize = shapeB.length == 3 ? shapeB[0] : 1;
        long batchSize = Math.max(aBatchSize, bBatchSize);

        long aBatchStride = stridesA.length == 3 ? stridesA[0] : 0;
        long bBatchStride = stridesB.length == 3 ? stridesB[0] : 0;
        long cBatchStride = resultStrides.length == 3 ? resultStrides[0] : 0;

        // TODO: MAKE THIS RESPECT STRIDES
        if (batchSize == 1) {
            matmul(transposeA,
                    transposeB,
                    m, n, k,
                    a,
                    0, lda,
                    b,
                    0, ldb,
                    result,
                    n,
                    0);
        } else {
            for (long i = 0; i < batchSize; i++) {
                matmul(transposeA,
                        transposeB,
                        m, n, k,
                        a,
                        (i % aBatchSize) * aBatchStride,
                        lda,
                        b,
                        (i % bBatchSize) * bBatchStride,
                        ldb,
                        result,
                        n,
                        i * cBatchStride
                );
            }
        }

        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shapeA = a.getShape();
        long[] shapeB = b.getShape();
        Validator.assertTrue(shapeA.length == 2 || shapeA.length == 3, "Only 2D or 3D matrices are supported");
        Validator.assertTrue(shapeB.length == 2 || shapeB.length == 3, "Only 2D or 3D matrices are supported");
        OptionBundle options = ctx.getOptionBundle();
        boolean transposeA = options.getOrDefault("transposeA", false);
        boolean transposeB = options.getOrDefault("transposeB", false);
        if (transposeA) {
            if (shapeA.length == 2) {
                shapeA = new long[]{shapeA[shapeA.length - 1], shapeA[shapeA.length - 2]};
            } else {
                shapeA = new long[]{shapeA[0], shapeA[shapeA.length - 1], shapeA[shapeA.length - 2]};
            }
        }
        if (transposeB) {
            if (shapeB.length == 2) {
                shapeB = new long[]{shapeB[shapeB.length - 1], shapeB[shapeB.length - 2]};
            } else {
                shapeB = new long[]{shapeB[0], shapeB[shapeB.length - 1], shapeB[shapeB.length - 2]};
            }
        }
        Validator.assertTrue(shapeA[shapeA.length - 1] == shapeB[shapeB.length - 2], "Matrix multiplication shape mismatch: " + ShapeUtils.toString(shapeA) + " x " + ShapeUtils.toString(shapeB));
        Validator.assertTrue(a.getDataType().isNumeric(), "Data type of A is not numeric");
        Validator.assertTrue(b.getDataType().isNumeric(), "Data type of B is not numeric");

        if (shapeA.length == 3 && shapeB.length == 3) {
            Validator.assertTrue(shapeA[0] == shapeB[0] || shapeA[0] == 1 || shapeB[0] == 1, "Batch size mismatch");
        }

        Validator.assertTrue(ShapeUtils.shapeFitsInInt(shapeA), "Shape of A is too large, no dimension must exceed Integer.MAX_VALUE");
        Validator.assertTrue(ShapeUtils.shapeFitsInInt(shapeB), "Shape of B is too large, no dimension must exceed Integer.MAX_VALUE");
        long[] resultShape = ShapeUtils.matrixMultiplyShape(shapeA, shapeB);
        DataType aDataType = a.getDataType();
        DataType bDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(aDataType, bDataType);
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

        ITensor aValue = a.getValue();
        ITensor bValue = b.getValue();

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
                dLdW = upstreamGradient.matmul(bValue, false, !transposeB);
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
                dLdW = bValue.matmul(upstreamGradient, transposeB, true);
            }
            // if we are performing a 3d matrix multiplication and 'a' is 3d with batch size 1,
            // or 'a' is 2d, then we need to sum the gradients over the batch dimension
            if (aValue.getShape().length == 3 || bValue.getShape().length == 3) {
                if (aValue.getShape().length == 3 && aValue.getShape()[0] == 1) {
                    dLdW = dLdW.reduceSum(0, true);
                } else if (aValue.getShape().length == 2) {
                    dLdW = dLdW.reduceSum(0, false);
                }
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
                dLdX = aValue.matmul(upstreamGradient, !transposeA, false);
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
                dLdX = upstreamGradient.matmul(aValue, true, transposeA);
            }
            // if we are performing a 3d matrix multiplication and 'b' is 3d with batch size 1,
            // or 'b' is 2d, then we need to sum the gradients over the batch dimension
            if (aValue.getShape().length == 3 || bValue.getShape().length == 3) {
                if (bValue.getShape().length == 3 && bValue.getShape()[0] == 1) {
                    dLdX = dLdX.reduceSum(0, true);
                } else if (bValue.getShape().length == 2) {
                    dLdX = dLdX.reduceSum(0, false);
                }
            }
            b.accumulateGradient(dLdX);
        }
    }

    private static void matmul(
            boolean transposeA,
            boolean transposeB,
            long m, long n, long k,
            ITensor a,
            long aOffset,
            long lda,
            ITensor b,
            long bOffset,
            long ldb,
            ITensor c,
            long ldc,
            long cOffset) {
        if (c.getDataType().isFloatingPoint()) {
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
                        double cValue = c.getAsDoubleFlat(cOffset + cIdx);
                        c.setByDoubleFlat(cValue + a.getAsDoubleFlat(aOffset + aIdx) * b.getAsDoubleFlat(bOffset + bIdx), cOffset + cIdx);
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
                        long cValue = c.getAsLongFlat(cOffset + cIdx);
                        c.setByLongFlat(cValue + a.getAsLongFlat(aOffset + aIdx) * b.getAsLongFlat(bOffset + bIdx), cOffset + cIdx);
                    }
                }
            }
        }
    }
}
