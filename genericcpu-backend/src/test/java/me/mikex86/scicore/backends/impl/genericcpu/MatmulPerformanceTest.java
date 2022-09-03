package me.mikex86.scicore.backends.impl.genericcpu;

import static me.mikex86.scicore.backend.impl.genericcpu.jni.MatmulJNI.*;

import me.mikex86.scicore.nativelib.LibraryLoader;
import org.junit.jupiter.api.Test;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.system.jemalloc.JEmalloc;

import java.nio.FloatBuffer;

public class MatmulPerformanceTest {

    static {
        LibraryLoader.loadLibrary("scicore_genericcpu");
    }

    @Test
    public void testMatmulPerformance() {
        long alphaPtr = JEmalloc.nje_malloc(4);
        MemoryUtil.memPutFloat(alphaPtr, 1.0f);
        int size = 4096;
        for (int i = 0; i < 1000; i++) {
            FloatBuffer a = JEmalloc.je_malloc(size * size * 4).asFloatBuffer();
            FloatBuffer b = JEmalloc.je_malloc(size * size * 4).asFloatBuffer();
            FloatBuffer c = JEmalloc.je_malloc(size * size * 4).asFloatBuffer();
            // fill random
            for (int j = 0; j < a.remaining(); j++) {
                a.put(j, (float) Math.random());
                b.put(j, (float) Math.random());
            }
            long start = System.nanoTime();
            matmul(OP_NONE, OP_NONE, size, size, size,
                    alphaPtr, MemoryUtil.memAddress(a), DATA_TYPE_FLOAT32, size,
                    alphaPtr, MemoryUtil.memAddress(b), DATA_TYPE_FLOAT32, size,
                    MemoryUtil.memAddress(c), DATA_TYPE_FLOAT32, size);
            long end = System.nanoTime();
            long nFlops = 2L * size * size * size;
            double tflops = (nFlops / ((end - start) / 1e9)) / 1e12;
            System.out.println("cuSgemm: " + tflops + " TFLOPS");
            JEmalloc.je_free(a);
            JEmalloc.je_free(b);
            JEmalloc.je_free(c);
        }
        JEmalloc.nje_free(alphaPtr);
    }
}
