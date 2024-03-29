package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.OptionBundle;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.TensorContentUtils;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import static me.mikex86.scicore.backend.impl.genericcpu.jni.MatmulJNI.*;

public class GenCPUMatMulOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final GenCPUBackend backend;

    public GenCPUMatMulOp(@NotNull GenCPUBackend backend) {
        this.backend = backend;
    }

    private static final Executor batchExecutor = Executors.newCachedThreadPool();

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

        DirectMemoryHandle aPtr = backend.getDirectMemoryManager().ensureDirect(a);
        DirectMemoryHandle bPtr = backend.getDirectMemoryManager().ensureDirect(b);

        // Because we are still using BLAS signatures that were specified in the 1970s, we unfortunately
        // cannot support arbitrary strides. Whenever we encounter a tensor with strides that are not
        // either the last two dims transposed or standard contiguous, we need to make a copy of the
        // tensor with standard contiguous strides. This is horrible, but this is also how PyTorch does
        // it, so this will my excuse to not feel bad about it.
        if (!stridesSupportedBySgemm(a.getShape(), a.getStrides())) {
            DirectMemoryHandle prevAPtr = aPtr;
            aPtr = TensorContentUtils.relayout(backend, aPtr, a.getShape(), a.getStrides(), ShapeUtils.makeStrides(a.getShape()), a.getDataType());
            if (prevAPtr.canFree()) {
                prevAPtr.free();
            }
            stridesA = ShapeUtils.makeStrides(shapeA);
        }

        if (!stridesSupportedBySgemm(b.getShape(), b.getStrides())) {
            DirectMemoryHandle prevBPtr = bPtr;
            bPtr = TensorContentUtils.relayout(backend, bPtr, b.getShape(), b.getStrides(), ShapeUtils.makeStrides(b.getShape()), b.getDataType());
            if (prevBPtr.canFree()) {
                prevBPtr.free();
            }
            stridesB = ShapeUtils.makeStrides(shapeB);
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

        // check if input has transposed strides
        if (isTransposedStrides(shapeA, stridesA)) {
            transposeA = !transposeA;

            // Swap the last two strides
            long[] newStrides = Arrays.copyOf(stridesA, stridesA.length); // new instance to not modify original tensor strides
            newStrides[newStrides.length - 1] = stridesA[stridesA.length - 2];
            newStrides[newStrides.length - 2] = stridesA[stridesA.length - 1];
            stridesA = newStrides;
        }

        if (isTransposedStrides(shapeB, stridesB)) {
            transposeB = !transposeB;

            // Swap the last two strides
            long[] newStrides = Arrays.copyOf(stridesB, stridesB.length); // new instance to not modify original tensor strides
            newStrides[newStrides.length - 1] = stridesB[stridesB.length - 2];
            newStrides[newStrides.length - 2] = stridesB[stridesB.length - 1];
            stridesB = newStrides;
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

        DirectMemoryHandle resultPtr = result.getContentsAsDirectMemory();

        long aBatchSize = shapeA.length == 3 ? shapeA[0] : 1;
        long bBatchSize = shapeB.length == 3 ? shapeB[0] : 1;
        long batchSize = Math.max(aBatchSize, bBatchSize);

        long aBatchStride = stridesA.length == 3 && shapeA[0] != 1 ? stridesA[0] : 0;
        long bBatchStride = stridesB.length == 3 && shapeB[0] != 1 ? stridesB[0] : 0;
        long cBatchStride = resultStrides.length == 3 && resultShape[0] != 1 ? resultStrides[0] : 0;

        if (batchSize == 1) {
            // Emulate strides with making lda and ldb the right stride values to get leading dim strides
            // change the layout between row major and column major to make the strides work out
            // This only works when one of the dimensions is 1, which we can make the column in the layout we choose.
            matmul(MATMUL_LAYOUT_ROW_MAJOR,
                    transposeA ? MATMUL_OP_TRANSPOSE : MATMUL_OP_NONE,
                    transposeB ? MATMUL_OP_TRANSPOSE : MATMUL_OP_NONE,
                    m, n, k,
                    aPtr.getNativePtr(),
                    getMatmulDataType(aDataType),
                    lda,
                    bPtr.getNativePtr(),
                    getMatmulDataType(bDataType),
                    ldb,
                    resultPtr.getNativePtr(),
                    getMatmulDataType(resultDataType),
                    n
            );
        } else {
            CountDownLatch latch = new CountDownLatch(Math.toIntExact(batchSize));
            for (long i = 0; i < batchSize; i++) {
                final long batchIndex = i;
                boolean finalTransposeA = transposeA;
                boolean finalTransposeB = transposeB;
                DirectMemoryHandle finalAPtr = aPtr;
                DirectMemoryHandle finalBPtr = bPtr;
                batchExecutor.execute(() -> {
                    matmul(MATMUL_LAYOUT_ROW_MAJOR,
                            finalTransposeA ? MATMUL_OP_TRANSPOSE : MATMUL_OP_NONE,
                            finalTransposeB ? MATMUL_OP_TRANSPOSE : MATMUL_OP_NONE,
                            m, n, k,
                            finalAPtr.offset(aDataType.getSizeOf((batchIndex % aBatchSize) * aBatchStride)).getNativePtr(),
                            getMatmulDataType(aDataType),
                            lda,
                            finalBPtr.offset(bDataType.getSizeOf((batchIndex % bBatchSize) * bBatchStride)).getNativePtr(),
                            getMatmulDataType(bDataType),
                            ldb,
                            resultPtr.offset(resultDataType.getSizeOf(batchIndex * cBatchStride)).getNativePtr(),
                            getMatmulDataType(resultDataType),
                            n
                    );
                    latch.countDown();
                });
            }
            try {
                latch.await();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
        if (aPtr.canFree()) {
            aPtr.free();
        }
        if (bPtr.canFree()) {
            bPtr.free();
        }

        return result;
    }

    private static boolean isTransposedStrides(long[] shape, long[] strides) {
        // note that dimensions in shape are permuted if indeed transposed, but strides are not!
        if (shape.length == 2) {
            return strides[0] == 1 && strides[1] == shape[0];
        } else if (shape.length == 3) {
            return strides[1] == 1 && strides[2] == shape[1];
        } else {
            throw new IllegalArgumentException("Only 2D and 3D matrices are supported");
        }
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

    private static boolean stridesSupportedBySgemm(long[] shape, long[] strides) {
        if (ShapeUtils.equals(ShapeUtils.makeStrides(shape), strides)) {
            return true;
        }
        if (strides.length == 2) {
            return strides[0] == 1 && strides[1] == shape[0];
        } else {
            return strides[0] == shape[1] * shape[2] && strides[1] == shape[2] && strides[2] == 1;
        }
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
}
