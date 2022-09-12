package me.mikex86.scicore;

public class JvmMatmulPerformanceTest {

    public static void main(String[] args) {
        ISciCore sciCore = new SciCore();
        sciCore.setBackend(SciCore.BackendType.JVM);

        // Matrix multiplication test
        ITensor a = sciCore.uniform(DataType.FLOAT32, 1000, 1000);
        ITensor b = sciCore.uniform(DataType.FLOAT32, 1000, 1000);

        long start = System.currentTimeMillis();
        ITensor c = sciCore.matmul(a, b);
        long end = System.currentTimeMillis();
        System.out.println("Matrix multiplication took " + (end - start) + "ms");
    }

}
