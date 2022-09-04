package me.mikex86.scicore.backend.impl.cuda;

import jcuda.Pointer;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.jcurand.curandGenerator;
import me.mikex86.scicore.DataType;
import me.mikex86.scicore.backend.impl.cuda.kernel.CudaKernel;
import me.mikex86.scicore.backend.impl.cuda.kernel.CudaKernelLaunchConfig;
import me.mikex86.scicore.backend.impl.cuda.kernel.KernelNameUtility;
import org.lwjgl.system.jemalloc.JEmalloc;

import java.nio.FloatBuffer;
import java.util.List;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.jcurand.JCurand.*;
import static jcuda.jcurand.JCurand.curandGenerateUniform;
import static jcuda.jcurand.curandRngType.CURAND_RNG_PSEUDO_DEFAULT;
import static me.mikex86.scicore.backend.impl.cuda.Validator.cuCheck;

public class KernelMatmulPerformanceTest {


    static {
        cuInit(0);

        CUdevice mainDevice = new CUdevice();
        cuCheck(cuDeviceGet(mainDevice, 0));

        // Create context
        {
            CUcontext ctx = new CUcontext();
            cuCheck(cuCtxCreate(ctx, 0, mainDevice));
        }
    }

    private static final CudaKernel matmulKernel = CudaKernel.loadClassPath("kernels/cuda/matmul.ptx", KernelNameUtility.getAllTypePermutations("matmul", List.of(DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64, DataType.FLOAT32, DataType.FLOAT64)));

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

            int xDimSize = size;
            int yDimSize = size;
            int blockDimX = 32, blockDimY = 32;
            int nBlocksX = Math.toIntExact((xDimSize + blockDimX - 1) / blockDimX);
            int nBlocksY = Math.toIntExact((yDimSize + blockDimY - 1) / blockDimY);

            // KERNEL_TEMPLATE void matmul(A *a, B *b, C *c, size_t m, size_t n, size_t k)
            matmulKernel.launchBlocking(
                    KernelNameUtility.getTypePermutation("matmul", DataType.FLOAT32, DataType.FLOAT32),
                    CudaKernelLaunchConfig.builder()
                            .blockDimX(blockDimX)
                            .blockDimY(blockDimY)
                            .gridDimX(nBlocksX)
                            .gridDimY(nBlocksY)
                            .parameters(
                                    Pointer.to(
                                            Pointer.to(aDevPtr),
                                            Pointer.to(bDevPtr),
                                            Pointer.to(cDevPtr),
                                            Pointer.to(new long[]{m}),
                                            Pointer.to(new long[]{n}),
                                            Pointer.to(new long[]{k})
                                    )
                            )
                            .build()
            );

            long end = System.nanoTime();
            double tflops = ((2 * m * n * k) / ((end - start) / 1e9)) / 1e12;
            cuMemFree(aDevPtr);
            cuMemFree(bDevPtr);
            cuMemFree(cDevPtr);
            System.out.println("matmul kernel took: " + (end - start) / 1e6 + "ms, " + tflops + " TFLOPS");
        }
        JEmalloc.je_free(factor);
    }

}
