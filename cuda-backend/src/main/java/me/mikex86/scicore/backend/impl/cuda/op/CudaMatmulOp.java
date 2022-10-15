package me.mikex86.scicore.backend.impl.cuda.op;

import jcuda.Pointer;
import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.LazyTensor;
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

import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.List;

import static jcuda.cudaDataType.CUDA_R_32F;
import static jcuda.cudaDataType.CUDA_R_64F;
import static jcuda.jcublas.JCublas2.cublasGemmEx_new;
import static jcuda.jcublas.cublasComputeType.*;
import static jcuda.jcublas.cublasGemmAlgo.CUBLAS_GEMM_DFALT_TENSOR_OP;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static me.mikex86.scicore.backend.impl.cuda.Validator.cublasCheck;

public class CudaMatmulOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final CudaBackend backend;

    @NotNull
    private final CudaKernel matmulKernel = CudaKernel.loadClassPath("kernels/cuda/matmul.ptx", KernelNameUtility.getAllTypePermutations("matmul", List.of(DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64, DataType.FLOAT32, DataType.FLOAT64)));

    public CudaMatmulOp(@NotNull CudaBackend backend) {
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
        DataType dataTypeA = a.getDataType();
        DataType dataTypeB = b.getDataType();
        DataType resultDataType = DataType.getLarger(dataTypeA, dataTypeB);

        CudaTensor result = new CudaTensor(this.backend, resultDataType, resultShape);
        CudaMemoryHandle resultMemoryHandle = result.getDataContainer().getDeviceMemoryHandle();

        int m = Math.toIntExact(opShapeA[0]),
                n = Math.toIntExact(opShapeB[1]),
                k = Math.toIntExact(opShapeA[1]);

        int lda = transposeA ? m : k;
        int ldb = transposeB ? k : n;

        CudaMemoryHandle aMemoryHandle = backend.getCudaMemoryManager().ensureOnDevice(a);
        CudaMemoryHandle bMemoryHandle = backend.getCudaMemoryManager().ensureOnDevice(b);

        // TODO: MAKE THIS RESPECT STRIDES
        if (a.getDataType() == DataType.FLOAT32 && b.getDataType() == DataType.FLOAT32) {
            // float32 by float32 multiplication
            // Swap A and B because cublasSgemm expects column-major matrices, and we have row-major.
            // Now, because B.T * A.T = C.T and A and B are already transposed when interpreted as column-major matrices, the result is C when interpreted as row-major.
            DirectMemoryHandle memoryHandle = backend.getDirectMemoryManager().alloc(1, DataType.FLOAT32);
            FloatBuffer factor = memoryHandle.asFloatBuffer();
            factor.put(1.0f);
            cublasCheck(cublasGemmEx_new(
                    CudaBackend.getCublasHandle(),
                    transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
                    n, m, k,
                    Pointer.to(factor),
                    bMemoryHandle.getDevicePointer(),
                    CUDA_R_32F,
                    lda,
                    aMemoryHandle.getDevicePointer(),
                    CUDA_R_32F,
                    ldb,
                    Pointer.to(factor),
                    resultMemoryHandle.getDevicePointer(),
                    CUDA_R_32F,
                    n,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DFALT_TENSOR_OP
            ));
            memoryHandle.free();
        } else if (a.getDataType() == DataType.FLOAT64 && b.getDataType() == DataType.FLOAT64) {
            // float64 by float64 multiplication
            // Also swap here because cublasDgemm expects column-major matrices, and we have row-major.
            DirectMemoryHandle memoryHandle = backend.getDirectMemoryManager().alloc(1, DataType.FLOAT64);
            DoubleBuffer factor = memoryHandle.asDoubleBuffer();
            factor.put(1.0);
            cublasCheck(cublasGemmEx_new(
                    CudaBackend.getCublasHandle(),
                    transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
                    n, m, k,
                    Pointer.to(factor),
                    bMemoryHandle.getDevicePointer(),
                    CUDA_R_64F,
                    lda,
                    aMemoryHandle.getDevicePointer(),
                    CUDA_R_64F,
                    ldb,
                    Pointer.to(factor),
                    resultMemoryHandle.getDevicePointer(),
                    CUDA_R_64F,
                    n,
                    CUBLAS_COMPUTE_64F,
                    CUBLAS_GEMM_DFALT_TENSOR_OP
            ));
            memoryHandle.free();
        } else {
            // Fallback kernel for every other data type combination not supported by cublas
            int xDimSize = (int) shapeA[0];
            int yDimSize = (int) shapeA[1];
            int blockDimX = 32, blockDimY = 32;
            int nBlocksX = Math.toIntExact((xDimSize + blockDimX - 1) / blockDimX);
            int nBlocksY = Math.toIntExact((yDimSize + blockDimY - 1) / blockDimY);

            // KERNEL_TEMPLATE void matmul(A *a, B *b, C *c, size_t m, size_t n, size_t k)
            this.matmulKernel.launchBlocking(
                    KernelNameUtility.getTypePermutation("matmul", dataTypeA, dataTypeB),
                    CudaKernelLaunchConfig.builder()
                            .blockDimX(blockDimX)
                            .blockDimY(blockDimY)
                            .gridDimX(nBlocksX)
                            .gridDimY(nBlocksY)
                            .parameters(
                                    Pointer.to(
                                            Pointer.to(aMemoryHandle.getDevicePointer()),
                                            Pointer.to(bMemoryHandle.getDevicePointer()),
                                            Pointer.to(result.getDataContainer().getDeviceMemoryHandle().getDevicePointer()),
                                            Pointer.to(new long[]{m}),
                                            Pointer.to(new long[]{n}),
                                            Pointer.to(new long[]{k})
                                    )
                            )
                            .build()
            );

        }

        if (aMemoryHandle.canFree()) // can free will be false if the memory was already on the device
            aMemoryHandle.free();

        if (bMemoryHandle.canFree()) // can free will be false if the memory was already on the device
            bMemoryHandle.free();

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
