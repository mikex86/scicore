package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.backend.impl.genericcpu.GenCPUTensor;
import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.OptionBundle;
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
        long[] shapeA = a.getShape();
        long[] shapeB = b.getShape();
        long[] stridesA = a.getStrides();
        long[] stridesB = b.getStrides();

        Validator.assertTrue(shapeB.length == 2, "Only 2D matrices are supported");
        Validator.assertTrue(shapeA.length == 2, "Only 2D matrices are supported");
        OptionBundle options = ctx.getOptionBundle();
        boolean transposeA = options.getOrDefault("transposeA", false);
        boolean transposeB = options.getOrDefault("transposeB", false);
        if (!(stridesA[0] == shapeA[1] && stridesA[1] == 1)) {
            throw new IllegalArgumentException("Invalid strides for matrix A"); // TODO: Support strides
        }
        if (!(stridesB[0] == shapeB[1] && stridesB[1] == 1)) {
            throw new IllegalArgumentException("Invalid strides for matrix B"); // TODO: Support strides
        }
        long[] opShapeA, opShapeB;
        if (transposeA) {
            opShapeA = new long[]{shapeA[1], shapeA[0]};
        } else {
            opShapeA = shapeA;
        }
        if (transposeB) {
            opShapeB = new long[]{shapeB[1], shapeB[0]};
        } else {
            opShapeB = shapeB;
        }
        Validator.assertTrue(opShapeA[1] == opShapeB[0], "Shape mismatch. A.shape[1] != B.shape[0]");
        Validator.assertTrue(a.getDataType().isNumeric(), "Data type of A is not numeric");
        Validator.assertTrue(b.getDataType().isNumeric(), "Data type of B is not numeric");
        long[] resultShape = ShapeUtils.matrixMultiplyShape(opShapeA, opShapeB);
        DataType aDataType = a.getDataType();
        DataType bDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(aDataType, bDataType);

        GenCPUTensor result = new GenCPUTensor(this.backend, resultDataType, resultShape);

        int m = Math.toIntExact(opShapeA[0]),
                n = Math.toIntExact(opShapeB[1]),
                k = Math.toIntExact(opShapeA[1]);

        int lda = transposeA ? m : k;
        int ldb = transposeB ? k : n;

        DirectMemoryHandle aPtr = backend.getDirectMemoryManager().ensureDirect(a);
        DirectMemoryHandle bPtr = backend.getDirectMemoryManager().ensureDirect(b);

        // TODO: MAKE THIS RESPECT STRIDES
        matmul(transposeA ? OP_TRANSPOSE : OP_NONE,
                transposeB ? OP_TRANSPOSE : OP_NONE,
                m, n, k,
                aPtr.getNativePtr(),
                getMatmulDataType(aDataType),
                lda,
                bPtr.getNativePtr(),
                getMatmulDataType(bDataType),
                ldb,
                result.getDataContainer().getMemoryHandle().getNativePtr(),
                getMatmulDataType(resultDataType),
                n
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
        Validator.assertTrue(ShapeUtils.shapeFitsInInt(shapeA), "Shape of A is too large, no dimension must exceed Integer.MAX_VALUE");
        Validator.assertTrue(ShapeUtils.shapeFitsInInt(shapeB), "Shape of B is too large, no dimension must exceed Integer.MAX_VALUE");
        long[] resultShape = ShapeUtils.matrixMultiplyShape(shapeA, shapeB);
        DataType aDataType = a.getDataType();
        DataType bDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(aDataType, bDataType);
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
        OptionBundle options = ctx.getOptionBundle();
        boolean transposeA = options.getOrDefault("transposeA", false);
        boolean transposeB = options.getOrDefault("transposeB", false);

        // TODO: OPTIMIZE

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
}
