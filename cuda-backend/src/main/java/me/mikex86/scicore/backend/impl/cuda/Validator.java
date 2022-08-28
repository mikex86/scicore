package me.mikex86.scicore.backend.impl.cuda;


import static jcuda.driver.JCudaDriver.cuGetErrorName;

public class Validator {

    public static void cuCheck(int result) {
        if (result != 0) {
            String[] errorName = new String[1];
            cuGetErrorName(result, errorName);
            throw new RuntimeException("CUDA error: " + errorName[0]);
        }
    }

}
