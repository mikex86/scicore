package me.mikex86.scicore.backend.impl.genericcpu.jni;

import me.mikex86.scicore.nativelib.LibraryLoader;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.system.jemalloc.JEmalloc;

import java.nio.DoubleBuffer;

import static me.mikex86.scicore.backend.impl.genericcpu.jni.MatmulJNI.*;

public class DoubleMatmulPerformanceTest {

    static {
        LibraryLoader.loadLibrary("scicore_genericcpu");
    }

    public static void main(String[] args) {
        long alphaPtr = JEmalloc.nje_malloc(8);
        MemoryUtil.memPutDouble(alphaPtr, 1.0f);
        int size = 1024;
        for (int i = 0; i < 1000; i++) {
            DoubleBuffer a = JEmalloc.je_malloc(size * size * 8).asDoubleBuffer();
            DoubleBuffer b = JEmalloc.je_malloc(size * size * 8).asDoubleBuffer();
            DoubleBuffer c = JEmalloc.je_malloc(size * size * 8).asDoubleBuffer();
            // fill random
            for (int j = 0; j < a.remaining(); j++) {
                a.put(j, Math.random());
                b.put(j, Math.random());
            }
            long start = System.nanoTime();
            matmul(MATMUL_OP_NONE, MATMUL_OP_NONE, size, size, size,
                    alphaPtr, MemoryUtil.memAddress(a), MATMUL_DATA_TYPE_FLOAT64, size,
                    alphaPtr, MemoryUtil.memAddress(b), MATMUL_DATA_TYPE_FLOAT64, size,
                    MemoryUtil.memAddress(c), MATMUL_DATA_TYPE_FLOAT64, size);
            long end = System.nanoTime();
            long nFlops = 2L * size * size * size;
            double tflops = (nFlops / ((end - start) / 1e9)) / 1e12;
            System.out.println("cblas_sgemm took " + (end - start) / 1e6 + " ms, " + tflops + " TFLOPS");
            JEmalloc.je_free(a);
            JEmalloc.je_free(b);
            JEmalloc.je_free(c);
        }
        JEmalloc.nje_free(alphaPtr);
    }
}
