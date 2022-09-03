package me.mikex86.scicore.backend.impl.cuda;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.jcurand.curandGenerator;
import org.junit.jupiter.api.Test;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.system.jemalloc.JEmalloc;

import java.nio.FloatBuffer;

import static jcuda.cudaDataType.CUDA_R_32F;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.jcublas.JCublas2.cublasGemmEx_new;
import static jcuda.jcublas.cublasComputeType.CUBLAS_COMPUTE_32F;
import static jcuda.jcublas.cublasGemmAlgo.CUBLAS_GEMM_DFALT_TENSOR_OP;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcurand.JCurand.*;
import static jcuda.jcurand.curandRngType.CURAND_RNG_PSEUDO_DEFAULT;
import static me.mikex86.scicore.backend.impl.cuda.Validator.cuCheck;
import static me.mikex86.scicore.backend.impl.cuda.Validator.cublasCheck;


public class CublasMatmulPerformanceTest {

    public static void main(String[] args) {
        FloatBuffer factor = JEmalloc.je_malloc(Float.BYTES).asFloatBuffer();
        factor.put(1.0f);
        int size = 1000;

        curandGenerator gen = new curandGenerator();
        curandCreateGenerator(gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234);

        for (int i = 0; i < 1000; i++) {

            CUdeviceptr aDevPtr = new CUdeviceptr();
            cuMemAlloc(aDevPtr, size * size * 4);
            CUdeviceptr bDevPtr = new CUdeviceptr();
            cuMemAlloc(bDevPtr, size * size * 4);
            CUdeviceptr cDevPtr = new CUdeviceptr();
            cuMemAlloc(cDevPtr, size * size * 4);

            // fill random
            curandGenerateUniform(gen, aDevPtr, size * size);
            curandGenerateUniform(gen, bDevPtr, size * size);

            long start = System.nanoTime();
            // cuSgemm
            int m = size, n = size, k = size;
            cublasCheck(cublasGemmEx_new(
                    CudaBackend.getCublasHandle(),
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    m, n, k,
                    Pointer.to(factor),
                    aDevPtr,
                    CUDA_R_32F,
                    m,
                    bDevPtr,
                    CUDA_R_32F,
                    k,
                    Pointer.to(factor),
                    cDevPtr,
                    CUDA_R_32F,
                    m,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DFALT_TENSOR_OP
            ));
            long end = System.nanoTime();
            double tflops = ((2 * m * n * k) / ((end - start) / 1e9)) / 1e12;
            System.out.println("cuSgemm: " + tflops + " TFLOPS");

            cuMemFree(aDevPtr);
            cuMemFree(bDevPtr);
            cuMemFree(cDevPtr);
        }
        JEmalloc.je_free(factor);
    }
}
