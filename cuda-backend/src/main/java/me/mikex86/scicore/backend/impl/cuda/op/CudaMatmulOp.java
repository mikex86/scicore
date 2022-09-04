package me.mikex86.scicore.backend.impl.cuda.op;

import jcuda.Pointer;
import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.View;
import me.mikex86.scicore.backend.impl.cuda.CudaBackend;
import me.mikex86.scicore.backend.impl.cuda.CudaTensor;
import me.mikex86.scicore.backend.impl.cuda.memory.CudaMemoryHandle;
import me.mikex86.scicore.backend.impl.cuda.memory.IMemoryHandle;
import me.mikex86.scicore.op.Graph;
import me.mikex86.scicore.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import me.mikex86.scicore.utils.ViewUtils;
import org.jetbrains.annotations.NotNull;
import org.lwjgl.system.jemalloc.JEmalloc;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;

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

    public CudaMatmulOp(@NotNull CudaBackend backend) {
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
        DataType ownDataType = a.getDataType();
        DataType otherDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(ownDataType, otherDataType);

        CudaTensor result = new CudaTensor(this.backend, resultDataType, resultShape);
        CudaMemoryHandle resultMemoryHandle = result.getDataContainer().getDeviceMemoryHandle();

        {
            // Swap A and B because cublasSgemm expects column-major matrices, and we have row-major.
            // Now, because B.T * A.T = C.T and A and B are already transposed when interpreted as column-major matrices, the result is C when interpreted as row-major.
            ITensor tmp = a;
            a = b;
            b = tmp;
        }

        int m = (int) shape[0]; // rows of A
        int k = (int) shape[1]; // columns of A and rows of B
        int n = (int) otherShape[1]; // columns of B

        // TODO: HANDLE OTHER DATA TYPES

        IMemoryHandle aDevicePtr, bDevicePtr;
        boolean aIsCopy = false, bIsCopy = false;
        {
            if (a instanceof CudaTensor aCudaTensor) {
                aDevicePtr = aCudaTensor.getDataContainer().getDeviceMemoryHandle();
            } else if (a instanceof View view && ViewUtils.getViewed(view) instanceof CudaTensor aCudaTensor) {
                long offset = aCudaTensor.getDataType().getSizeOf(ViewUtils.getTotalOffset(view));
                aDevicePtr = aCudaTensor.getDataContainer().getDeviceMemoryHandle().offset(offset);
            } else {
                aDevicePtr = backend.getMemoryManager().copyToDevice(a);
                aIsCopy = true;
            }

            if (b instanceof CudaTensor bCudaTensor) {
                bDevicePtr = bCudaTensor.getDataContainer().getDeviceMemoryHandle();
            } else if (b instanceof View view && ViewUtils.getViewed(view) instanceof CudaTensor bCudaTensor) {
                long offset = ViewUtils.getTotalOffset(view);
                bDevicePtr = bCudaTensor.getDataContainer().getDeviceMemoryHandle().offset(offset);
            } else {
                bDevicePtr = backend.getMemoryManager().copyToDevice(b);
                bIsCopy = true;
            }
        }

        if (a.getDataType() == DataType.FLOAT32 && b.getDataType() == DataType.FLOAT32) {
            // float32 by float32 multiplication
            FloatBuffer factor = backend.getDirectMemoryManager().allocFloatBuffer(1);
            factor.put(1.0f);
            cublasCheck(cublasGemmEx_new(
                    CudaBackend.getCublasHandle(),
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    m, n, k,
                    Pointer.to(factor),
                    aDevicePtr.getPointer(),
                    CUDA_R_32F,
                    m,
                    bDevicePtr.getPointer(),
                    CUDA_R_32F,
                    k,
                    Pointer.to(factor),
                    resultMemoryHandle.getPointer(),
                    CUDA_R_32F,
                    m,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DFALT_TENSOR_OP
            ));
            backend.getDirectMemoryManager().free(factor);
        } else if (a.getDataType() == DataType.FLOAT64 && b.getDataType() == DataType.FLOAT64) {
            // float64 by float64 multiplication
            DoubleBuffer factor = backend.getDirectMemoryManager().allocDoubleBuffer(1);
            factor.put(1.0);
            cublasCheck(cublasGemmEx_new(
                    CudaBackend.getCublasHandle(),
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    m, n, k,
                    Pointer.to(factor),
                    aDevicePtr.getPointer(),
                    CUDA_R_64F,
                    m,
                    bDevicePtr.getPointer(),
                    CUDA_R_64F,
                    k,
                    Pointer.to(factor),
                    resultMemoryHandle.getPointer(),
                    CUDA_R_64F,
                    m,
                    CUBLAS_COMPUTE_64F,
                    CUBLAS_GEMM_DFALT_TENSOR_OP
            ));
            backend.getDirectMemoryManager().free(factor);
        } else {
            // TODO: HANDLE OTHER DATA TYPES
            throw new RuntimeException("TODO: Unsupported data type combination");
        }

        if (aIsCopy)
            aDevicePtr.free();

        if (bIsCopy)
            bDevicePtr.free();

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
        long[] resultShape = ShapeUtils.matrixMultiplyShape(shape, otherShape);
        DataType ownDataType = a.getDataType();
        DataType otherDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(ownDataType, otherDataType);
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

        // TODO: HANDLE VIEWS PROPERLY
        if (a.requiresGradients()) {
            ITensor dLdW;
            if (upstreamGradient instanceof CudaTensor upstreamTensor && b.getValue() instanceof CudaTensor bCudaTensor) {
                /*
                 This is equivalent to:
                  dLdW = upstreamGradient.matmul(b.getValue().transpose());
                 */

                CudaMemoryHandle upstreamDevPtr = upstreamTensor.getDataContainer().getDeviceMemoryHandle();
                CudaMemoryHandle bDevPtr = bCudaTensor.getDataContainer().getDeviceMemoryHandle();
                DataType resultDataType = DataType.getLarger(upstreamGradient.getDataType(), bCudaTensor.getDataType());
                long[] resultShape = ShapeUtils.matrixMultiplyShape(upstreamGradient.getShape(), bCudaTensor.getShape());
                dLdW = this.backend.createTensor(resultDataType, resultShape);
                CudaMemoryHandle dLdWDevPtr = ((CudaTensor) dLdW).getDataContainer().getDeviceMemoryHandle();
                int m = Math.toIntExact(resultShape[0]);
                int n = Math.toIntExact(resultShape[1]);
                int k = Math.toIntExact(upstreamGradient.getShape()[1]);

                // Swap upstreamGradient and b because cublasSgemm expects column-major matrices, and we have row-major.
                if (upstreamGradient.getDataType() == DataType.FLOAT32 && bCudaTensor.getDataType() == DataType.FLOAT32) {
                    // float32 by float32 multiplication
                    FloatBuffer factor = backend.getDirectMemoryManager().allocFloatBuffer(1);
                    factor.put(1.0f);
                    cublasCheck(cublasGemmEx_new(
                            CudaBackend.getCublasHandle(),
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            n, m, k,
                            Pointer.to(factor),
                            bDevPtr.getPointer(),
                            CUDA_R_32F,
                            k,
                            upstreamDevPtr.getPointer(),
                            CUDA_R_32F,
                            m,
                            Pointer.to(factor),
                            dLdWDevPtr.getPointer(),
                            CUDA_R_32F,
                            n,
                            CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DFALT_TENSOR_OP
                    ));
                    backend.getDirectMemoryManager().free(factor);
                } else if (upstreamGradient.getDataType() == DataType.FLOAT64 && bCudaTensor.getDataType() == DataType.FLOAT64) {
                    // float64 by float64 multiplication
                    DoubleBuffer factor = backend.getDirectMemoryManager().allocDoubleBuffer(1);
                    factor.put(1.0);
                    cublasCheck(cublasGemmEx_new(
                            CudaBackend.getCublasHandle(),
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            n, m, k,
                            Pointer.to(factor),
                            bDevPtr.getPointer(),
                            CUDA_R_64F,
                            k,
                            upstreamDevPtr.getPointer(),
                            CUDA_R_64F,
                            m,
                            Pointer.to(factor),
                            dLdWDevPtr.getPointer(),
                            CUDA_R_64F,
                            n,
                            CUBLAS_COMPUTE_64F,
                            CUBLAS_GEMM_DFALT_TENSOR_OP
                    ));
                    backend.getDirectMemoryManager().free(factor);
                } else {
                    throw new UnsupportedOperationException("TODO: Unsupported data type combination");
                }
            } else {
                dLdW = upstreamGradient.matmul(b.getValue().transpose());
            }
            a.accumulateGradient(dLdW);
        }

        if (b.requiresGradients()) {
            ITensor dLdX;
            if (upstreamGradient instanceof CudaTensor upstreamTensor && a.getValue() instanceof CudaTensor aCudaTensor) {
                /*
                 This is equivalent to:
                  dLdX = a.getValue().transpose().matmul(upstreamGradient);
                 */
                CudaMemoryHandle upstreamDevPtr = upstreamTensor.getDataContainer().getDeviceMemoryHandle();
                CudaMemoryHandle aDevPtr = aCudaTensor.getDataContainer().getDeviceMemoryHandle();
                DataType resultDataType = DataType.getLarger(upstreamGradient.getDataType(), aCudaTensor.getDataType());
                long[] resultShape = ShapeUtils.matrixMultiplyShape(aCudaTensor.getShape(), upstreamGradient.getShape());

                dLdX = this.backend.createTensor(resultDataType, resultShape);
                CudaMemoryHandle dLdXDevPtr = ((CudaTensor) dLdX).getDataContainer().getDeviceMemoryHandle();
                int m = Math.toIntExact(resultShape[0]);
                int n = Math.toIntExact(resultShape[1]);
                int k = Math.toIntExact(aCudaTensor.getShape()[1]);

                // Swap upstreamGradient and a because cublasSgemm expects column-major matrices, and we have row-major.
                if (upstreamGradient.getDataType() == DataType.FLOAT32 && aCudaTensor.getDataType() == DataType.FLOAT32) {
                    // float32 by float32 multiplication
                    FloatBuffer factor = backend.getDirectMemoryManager().allocFloatBuffer(1);
                    factor.put(1.0f);
                    cublasCheck(cublasGemmEx_new(
                            CudaBackend.getCublasHandle(),
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            n, m, k,
                            Pointer.to(factor),
                            upstreamDevPtr.getPointer(),
                            CUDA_R_32F,
                            k,
                            aDevPtr.getPointer(),
                            CUDA_R_32F,
                            m,
                            Pointer.to(factor),
                            dLdXDevPtr.getPointer(),
                            CUDA_R_32F,
                            n,
                            CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DFALT_TENSOR_OP
                    ));
                    backend.getDirectMemoryManager().free(factor);
                } else if (upstreamGradient.getDataType() == DataType.FLOAT64 && aCudaTensor.getDataType() == DataType.FLOAT64) {
                    // float64 by float64 multiplication
                    DoubleBuffer factor = backend.getDirectMemoryManager().allocDoubleBuffer(1);
                    factor.put(1.0);
                    cublasCheck(cublasGemmEx_new(
                            CudaBackend.getCublasHandle(),
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            n, m, k,
                            Pointer.to(factor),
                            upstreamDevPtr.getPointer(),
                            CUDA_R_64F,
                            k,
                            aDevPtr.getPointer(),
                            CUDA_R_64F,
                            m,
                            Pointer.to(factor),
                            dLdXDevPtr.getPointer(),
                            CUDA_R_64F,
                            n,
                            CUBLAS_COMPUTE_64F,
                            CUBLAS_GEMM_DFALT_TENSOR_OP
                    ));
                    backend.getDirectMemoryManager().free(factor);
                } else {
                    throw new UnsupportedOperationException("TODO: Unsupported data type combination");
                }
            } else {
                dLdX = a.getValue().transpose().matmul(upstreamGradient);
            }
            b.accumulateGradient(dLdX);
        }
    }

}
