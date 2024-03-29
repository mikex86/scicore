package me.mikex86.scicore.backend.impl.cuda;

import static jcuda.driver.JCudaDriver.cuGetErrorName;
import static jcuda.jcublas.JCublas2.cublasGetStatusName;
import static jcuda.jcublas.cublasStatus.CUBLAS_STATUS_SUCCESS;
import static jcuda.nvrtc.JNvrtc.nvrtcGetErrorString;

public class Validator {

    public static void cuCheck(int result) {
        if (result != 0) {
            String[] errorName = new String[1];
            cuGetErrorName(result, errorName);
            throw new RuntimeException("CUDA error: " + errorName[0]);
        }
    }

    public static void cublasCheck(int result) {
        if (result != CUBLAS_STATUS_SUCCESS) {
            String errorName = cublasGetStatusName(result);
            throw new RuntimeException("cuBLAS error: " + errorName);
        }
    }

    public static void nvrtcCheck(int result) {
        if (result != 0) {
            String errorName = nvrtcGetErrorString(result);
            throw new RuntimeException("NVRTC error: " + errorName);
        }
    }


}
