package me.mikex86.scicore.backend.impl.cuda.op;

import jcuda.Pointer;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.backend.impl.cuda.CudaBackend;
import me.mikex86.scicore.backend.impl.cuda.CudaTensor;
import me.mikex86.scicore.backend.impl.cuda.kernel.CudaKernel;
import me.mikex86.scicore.backend.impl.cuda.kernel.CudaKernelLaunchConfig;
import me.mikex86.scicore.backend.impl.cuda.kernel.KernelNameUtility;
import me.mikex86.scicore.backend.impl.cuda.memory.CudaMemoryHandle;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.OptionBundle;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.awt.*;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.List;
import java.util.concurrent.CountDownLatch;

import static jcuda.cudaDataType.CUDA_R_32F;
import static jcuda.cudaDataType.CUDA_R_64F;
import static jcuda.jcublas.JCublas2.*;
import static jcuda.jcublas.cublasComputeType.*;
import static jcuda.jcublas.cublasGemmAlgo.*;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static me.mikex86.scicore.backend.impl.cuda.Validator.cublasCheck;

public class CudaMatmulOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final CudaBackend backend;

    @NotNull
    private final CudaKernel matmulKernel = CudaKernel.loadClassPath("kernels/cuda/matmul.ptx", KernelNameUtility.getAllTypePermutations("matmul", List.of(DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64, DataType.FLOAT32, DataType.FLOAT64)));

    @Nullable
    private DirectMemoryHandle float32AlphaHandle;

    @Nullable
    private DirectMemoryHandle float64AlphaHandle;

    public CudaMatmulOp(@NotNull CudaBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] rowMajorShapeA = a.getShape();
        long[] rowMajorShapeB = b.getShape();

        long[] stridesA = a.getStrides();
        long[] stridesB = b.getStrides();

        Validator.assertTrue(ShapeUtils.shapeFitsInInt(rowMajorShapeA), "Shape of A is too large, no dimension must exceed Integer.MAX_VALUE");
        Validator.assertTrue(ShapeUtils.shapeFitsInInt(rowMajorShapeB), "Shape of B is too large, no dimension must exceed Integer.MAX_VALUE");

        OptionBundle options = ctx.getOptionBundle();
        boolean transposeA = options.getOrDefault("transposeA", false);
        boolean transposeB = options.getOrDefault("transposeB", false);

        long[] opRowMajorShapeA;
        long[] opRowMajorShapeB;
        long[] resultShapeRowMajor;
        {
            Validator.assertTrue(rowMajorShapeA.length == 2 || rowMajorShapeA.length == 3, "Only 2D and 3D tensors are supported");
            Validator.assertTrue(rowMajorShapeB.length == 2 || rowMajorShapeB.length == 3, "Only 2D and 3D tensors are supported");

            // TODO: Support more complex strides
            if (stridesA[stridesA.length - 1] != 1 &&
                stridesA[stridesA.length - 1] == rowMajorShapeA[rowMajorShapeA.length - 1]
                && stridesA[stridesA.length - 2] == 1) {
                transposeA = !transposeA;
            }

            if (stridesB[stridesB.length - 1] != 1 &&
                stridesB[stridesB.length - 1] == rowMajorShapeB[rowMajorShapeB.length - 1]
                && stridesB[stridesB.length - 2] == 1) {
                transposeB = !transposeB;
            }

            if (transposeA) {
                if (rowMajorShapeA.length == 2) {
                    opRowMajorShapeA = new long[]{rowMajorShapeA[rowMajorShapeA.length - 1], rowMajorShapeA[rowMajorShapeA.length - 2]};
                } else {
                    opRowMajorShapeA = new long[]{rowMajorShapeA[0], rowMajorShapeA[rowMajorShapeA.length - 1], rowMajorShapeA[rowMajorShapeA.length - 2]};
                }
            } else {
                opRowMajorShapeA = rowMajorShapeA;
            }
            if (transposeB) {
                if (rowMajorShapeB.length == 2) {
                    opRowMajorShapeB = new long[]{rowMajorShapeB[rowMajorShapeB.length - 1], rowMajorShapeB[rowMajorShapeB.length - 2]};
                } else {
                    opRowMajorShapeB = new long[]{rowMajorShapeB[0], rowMajorShapeB[rowMajorShapeB.length - 1], rowMajorShapeB[rowMajorShapeB.length - 2]};
                }
            } else {
                opRowMajorShapeB = rowMajorShapeB;
            }

            Validator.assertTrue(opRowMajorShapeA[opRowMajorShapeA.length - 1] == opRowMajorShapeB[opRowMajorShapeB.length - 2], "Matrix multiplication shape mismatch: " + ShapeUtils.toString(opRowMajorShapeA) + " x " + ShapeUtils.toString(opRowMajorShapeB));
            Validator.assertTrue(a.getDataType().isNumeric(), "Data type of A is not numeric");
            Validator.assertTrue(b.getDataType().isNumeric(), "Data type of B is not numeric");

            if (opRowMajorShapeA.length == 3 && opRowMajorShapeB.length == 3) {
                Validator.assertTrue(opRowMajorShapeA[0] == opRowMajorShapeB[0] || opRowMajorShapeA[0] == 1 || opRowMajorShapeB[0] == 1, "Batch size mismatch");
            }

            resultShapeRowMajor = ShapeUtils.matrixMultiplyShape(opRowMajorShapeA, opRowMajorShapeB);
        }

        DataType aDataType = a.getDataType();
        DataType bDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(aDataType, bDataType);

        CudaTensor result = this.backend.createTensor(resultDataType, resultShapeRowMajor);

        long[] resultStrides = result.getStrides();

        CudaMemoryHandle aPtr = backend.getCudaMemoryManager().ensureOnDevice(a);
        CudaMemoryHandle bPtr = backend.getCudaMemoryManager().ensureOnDevice(b);
        CudaMemoryHandle resultPtr = result.getDataContainer().getDeviceMemoryHandle();

        long aBatchSize = rowMajorShapeA.length == 3 ? rowMajorShapeA[0] : 1;
        long bBatchSize = rowMajorShapeB.length == 3 ? rowMajorShapeB[0] : 1;
        long batchSize = Math.max(aBatchSize, bBatchSize);

        long aBatchStride = stridesA.length == 3 && opRowMajorShapeA[0] != 1 ? stridesA[0] : 0;
        long bBatchStride = stridesB.length == 3 && opRowMajorShapeB[0] != 1 ? stridesB[0] : 0;
        long cBatchStride = resultStrides.length == 3 && resultShapeRowMajor[0] != 1 ? resultStrides[0] : 0;

        long[] shapeColumnMajorA = new long[]{rowMajorShapeA[rowMajorShapeA.length - 1], rowMajorShapeA[rowMajorShapeA.length - 2]};
        long[] shapeColumnMajorB = new long[]{rowMajorShapeB[rowMajorShapeB.length - 1], rowMajorShapeB[rowMajorShapeB.length - 2]};

        // cublasSgemm expects column-major matrices, and we have row-major.
        // Now, because B.T * A.T = C.T and A and B are already transposed when interpreted as column-major matrices, the result is C when interpreted as row-major.
        if (a.getDataType() == DataType.FLOAT32 && b.getDataType() == DataType.FLOAT32) {
            if (float32AlphaHandle == null) {
                float32AlphaHandle = backend.getDirectMemoryManager().alloc(1, DataType.FLOAT32);
                FloatBuffer factor = float32AlphaHandle.asFloatBuffer();
                factor.put(1.0f);
            }
            if (!transposeA && !transposeB) {
                cublasCheck(cublasGemmStridedBatchedEx_new(
                        CudaBackend.getCublasHandle(),
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        (int) shapeColumnMajorB[0], (int) shapeColumnMajorA[1], (int) shapeColumnMajorB[1],
                        Pointer.to(float32AlphaHandle.asFloatBuffer()),
                        bPtr.getDevicePointer(),
                        CUDA_R_32F,
                        (int) shapeColumnMajorB[0],
                        (int) bBatchStride,
                        aPtr.getDevicePointer(),
                        CUDA_R_32F,
                        (int) shapeColumnMajorA[0],
                        (int) aBatchStride,
                        Pointer.to(float32AlphaHandle.asFloatBuffer()),
                        resultPtr.getDevicePointer(),
                        CUDA_R_32F,
                        (int) shapeColumnMajorB[0],
                        (int) cBatchStride,
                        (int) batchSize,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DFALT_TENSOR_OP
                ));
            } else if (!transposeA) {
                cublasCheck(cublasGemmStridedBatchedEx_new(
                        CudaBackend.getCublasHandle(),
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        (int) shapeColumnMajorB[1], (int) shapeColumnMajorA[1], (int) shapeColumnMajorB[0],
                        Pointer.to(float32AlphaHandle.asFloatBuffer()),
                        bPtr.getDevicePointer(),
                        CUDA_R_32F,
                        (int) shapeColumnMajorB[0], // lda = yStride = opShapeColumnMajorB[1] = shapeRowMajorB[0] ... Curse you 1970s Fortran indexing
                        (int) bBatchStride,
                        aPtr.getDevicePointer(),
                        CUDA_R_32F,
                        (int) shapeColumnMajorA[0],
                        (int) aBatchStride,
                        Pointer.to(float32AlphaHandle.asFloatBuffer()),
                        resultPtr.getDevicePointer(),
                        CUDA_R_32F,
                        (int) shapeColumnMajorB[1],
                        (int) cBatchStride,
                        (int) batchSize,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_ALGO0_TENSOR_OP
                ));
            } else if (!transposeB) {
                cublasCheck(cublasGemmStridedBatchedEx_new(
                        CudaBackend.getCublasHandle(),
                        CUBLAS_OP_N, CUBLAS_OP_T,
                        (int) shapeColumnMajorB[0], (int) shapeColumnMajorA[0], (int) shapeColumnMajorB[1],
                        Pointer.to(float32AlphaHandle.asFloatBuffer()),
                        bPtr.getDevicePointer(),
                        CUDA_R_32F,
                        (int) shapeColumnMajorB[0],
                        (int) bBatchStride,
                        aPtr.getDevicePointer(),
                        CUDA_R_32F,
                        (int) shapeColumnMajorA[0],
                        (int) aBatchStride,
                        Pointer.to(float32AlphaHandle.asFloatBuffer()),
                        resultPtr.getDevicePointer(),
                        CUDA_R_32F,
                        (int) shapeColumnMajorB[0],
                        (int) cBatchStride,
                        (int) batchSize,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_ALGO0_TENSOR_OP
                ));
            } else {
                cublasCheck(cublasGemmStridedBatchedEx_new(
                        CudaBackend.getCublasHandle(),
                        CUBLAS_OP_T, CUBLAS_OP_T,
                        (int) shapeColumnMajorB[1], (int) shapeColumnMajorA[0], (int) shapeColumnMajorB[0],
                        Pointer.to(float32AlphaHandle.asFloatBuffer()),
                        bPtr.getDevicePointer(),
                        CUDA_R_32F,
                        (int) shapeColumnMajorB[0],
                        (int) bBatchStride,
                        aPtr.getDevicePointer(),
                        CUDA_R_32F,
                        (int) shapeColumnMajorA[0],
                        (int) aBatchStride,
                        Pointer.to(float32AlphaHandle.asFloatBuffer()),
                        resultPtr.getDevicePointer(),
                        CUDA_R_32F,
                        (int) shapeColumnMajorB[1],
                        (int) cBatchStride,
                        (int) batchSize,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_ALGO0_TENSOR_OP
                ));
            }
        } else if (a.getDataType() == DataType.FLOAT64 && b.getDataType() == DataType.FLOAT64) {
            if (float64AlphaHandle == null) {
                float64AlphaHandle = backend.getDirectMemoryManager().alloc(1, DataType.FLOAT64);
                DoubleBuffer factor = float64AlphaHandle.asDoubleBuffer();
                factor.put(1.0);
            }
            if (!transposeA && !transposeB) {
                cublasCheck(cublasGemmStridedBatchedEx_new(
                        CudaBackend.getCublasHandle(),
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        (int) shapeColumnMajorB[0], (int) shapeColumnMajorA[1], (int) shapeColumnMajorB[1],
                        Pointer.to(float64AlphaHandle.asDoubleBuffer()),
                        bPtr.getDevicePointer(),
                        CUDA_R_64F,
                        (int) shapeColumnMajorB[0],
                        (int) bBatchStride,
                        aPtr.getDevicePointer(),
                        CUDA_R_64F,
                        (int) shapeColumnMajorA[0],
                        (int) aBatchStride,
                        Pointer.to(float64AlphaHandle.asDoubleBuffer()),
                        resultPtr.getDevicePointer(),
                        CUDA_R_64F,
                        (int) shapeColumnMajorB[0],
                        (int) cBatchStride,
                        (int) batchSize,
                        CUBLAS_COMPUTE_64F,
                        CUBLAS_GEMM_ALGO0_TENSOR_OP
                ));
            } else if (!transposeA) {
                cublasCheck(cublasGemmStridedBatchedEx_new(
                        CudaBackend.getCublasHandle(),
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        (int) shapeColumnMajorB[1], (int) shapeColumnMajorA[1], (int) shapeColumnMajorB[0],
                        Pointer.to(float64AlphaHandle.asDoubleBuffer()),
                        bPtr.getDevicePointer(),
                        CUDA_R_64F,
                        (int) shapeColumnMajorB[0],
                        (int) bBatchStride,
                        aPtr.getDevicePointer(),
                        CUDA_R_64F,
                        (int) shapeColumnMajorA[0],
                        (int) aBatchStride,
                        Pointer.to(float64AlphaHandle.asDoubleBuffer()),
                        resultPtr.getDevicePointer(),
                        CUDA_R_64F,
                        (int) shapeColumnMajorB[1],
                        (int) cBatchStride,
                        (int) batchSize,
                        CUBLAS_COMPUTE_64F,
                        CUBLAS_GEMM_ALGO0_TENSOR_OP
                ));
            } else if (!transposeB) {
                cublasCheck(cublasGemmStridedBatchedEx_new(
                        CudaBackend.getCublasHandle(),
                        CUBLAS_OP_N, CUBLAS_OP_T,
                        (int) shapeColumnMajorB[0], (int) shapeColumnMajorA[0], (int) shapeColumnMajorB[1],
                        Pointer.to(float64AlphaHandle.asDoubleBuffer()),
                        bPtr.getDevicePointer(),
                        CUDA_R_64F,
                        (int) shapeColumnMajorB[0],
                        (int) bBatchStride,
                        aPtr.getDevicePointer(),
                        CUDA_R_64F,
                        (int) shapeColumnMajorA[0],
                        (int) aBatchStride,
                        Pointer.to(float64AlphaHandle.asDoubleBuffer()),
                        resultPtr.getDevicePointer(),
                        CUDA_R_64F,
                        (int) shapeColumnMajorB[0],
                        (int) cBatchStride,
                        (int) batchSize,
                        CUBLAS_COMPUTE_64F,
                        CUBLAS_GEMM_ALGO0_TENSOR_OP
                ));
            } else {
                cublasCheck(cublasGemmStridedBatchedEx_new(
                        CudaBackend.getCublasHandle(),
                        CUBLAS_OP_T, CUBLAS_OP_T,
                        (int) shapeColumnMajorB[1], (int) shapeColumnMajorA[0], (int) shapeColumnMajorB[0],
                        Pointer.to(float64AlphaHandle.asDoubleBuffer()),
                        bPtr.getDevicePointer(),
                        CUDA_R_64F,
                        (int) shapeColumnMajorB[0],
                        (int) bBatchStride,
                        aPtr.getDevicePointer(),
                        CUDA_R_64F,
                        (int) shapeColumnMajorA[0],
                        (int) aBatchStride,
                        Pointer.to(float64AlphaHandle.asDoubleBuffer()),
                        resultPtr.getDevicePointer(),
                        CUDA_R_64F,
                        (int) shapeColumnMajorB[1],
                        (int) cBatchStride,
                        (int) batchSize,
                        CUBLAS_COMPUTE_64F,
                        CUBLAS_GEMM_ALGO0_TENSOR_OP
                ));
            }
        } else {
            // Fallback kernel for every other data type combination not supported by cublas
            int xDimSize = (int) resultShapeRowMajor[resultShapeRowMajor.length - 2];
            int yDimSize = (int) resultShapeRowMajor[resultShapeRowMajor.length - 1];
            int zDimSize = (int) batchSize;

            int xThreads = 32;
            int yThreads = 32;
            int zThreads = 1;
            int xBlocks = (xDimSize + xThreads - 1) / xThreads;
            int yBlocks = (yDimSize + yThreads - 1) / yThreads;
            int zBlocks = (zDimSize + zThreads - 1) / zThreads;

            int m = (int) opRowMajorShapeA[opRowMajorShapeA.length - 2];
            int n = (int) opRowMajorShapeB[opRowMajorShapeB.length - 1];
            int k = (int) opRowMajorShapeA[opRowMajorShapeA.length - 1];

            // KERNEL_TEMPLATE void matmul(A *a, B *b, C *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize)
            this.matmulKernel.launchBlocking(
                    KernelNameUtility.getTypePermutation("matmul", aDataType, bDataType),
                    CudaKernelLaunchConfig.builder()
                            .blockDimX(xThreads)
                            .blockDimY(yThreads)
                            .blockDimZ(zThreads)
                            .gridDimX(xBlocks)
                            .gridDimY(yBlocks)
                            .gridDimZ(zBlocks)
                            .parameters(
                                    Pointer.to(
                                            Pointer.to(aPtr.getDevicePointer()),
                                            Pointer.to(bPtr.getDevicePointer()),
                                            Pointer.to(resultPtr.getDevicePointer()),
                                            Pointer.to(new long[]{m}),
                                            Pointer.to(new long[]{n}),
                                            Pointer.to(new long[]{k}),
                                            Pointer.to(new int[]{transposeA ? 1 : 0}),
                                            Pointer.to(new int[]{transposeB ? 1 : 0}),
                                            Pointer.to(new long[]{aBatchStride}),
                                            Pointer.to(new long[]{bBatchStride}),
                                            Pointer.to(new long[]{cBatchStride}),
                                            Pointer.to(new long[]{batchSize})
                                    )
                            )
                            .build()
            );
        }
        if (aPtr.canFree()) // can free will be false if the memory was already on the device
            aPtr.free();

        if (bPtr.canFree()) // can free will be false if the memory was already on the device
            bPtr.free();

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

}
