package me.mikex86.scicore.tests;

import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.SciCore;
import me.mikex86.scicore.tensor.ITensor;

public class CudaMatmulTest {

    public static void main(String[] args) {
        SciCore sciCore = new SciCore();
        sciCore.addBackend(ISciCore.BackendType.CUDA);

        // transposeA = false, transposeB = false
//        ITensor a = sciCore.matrix(new float[][]{{1, 2}, {3, 4}, {5, 6}});
//        ITensor b = sciCore.matrix(new float[][]{{5, 6, 7, 8}, {9, 10, 11, 12}});
//        ITensor c = sciCore.matmul(a, b, false, false);

        // transposeA = false, transposeB = true
//        ITensor a = sciCore.matrix(new float[][]{{1, 2}, {3, 4}, {5, 6}});
//        ITensor b = sciCore.matrix(new float[][]{{5, 6}, {7, 8}, {9, 10}, {11, 12}});
//        ITensor c = sciCore.matmul(a, b, false, true);

        // transposeA = true, transposeB = false
//        ITensor a = sciCore.matrix(new float[][]{{1, 2, 3}, {4, 5, 6}});
//        ITensor b = sciCore.matrix(new float[][]{{5, 6, 7, 8}, {9, 10, 11, 12}});
//        ITensor c = sciCore.matmul(a, b, true, false);

        // transposeA = true, transposeB = true
//        ITensor a = sciCore.matrix(new float[][]{{1, 2, 3}, {4, 5, 6}});
//        ITensor b = sciCore.matrix(new float[][]{{5, 6}, {7, 8}, {9, 10}, {11, 12}});
//        ITensor c = sciCore.matmul(a, b, true, true);

        ITensor a = sciCore.matrix(new byte[][]{{1, 2}, {3, 4}});
        ITensor b = sciCore.matrix(new short[][]{{5, 6}, {7, 8}});
        ITensor c = sciCore.matmul(a, b, true, false);
        c.getAsFloatFlat(0);
        System.out.println(c);
    }

}
