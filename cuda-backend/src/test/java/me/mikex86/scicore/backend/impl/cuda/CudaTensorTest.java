package me.mikex86.scicore.backend.impl.cuda;

import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.SciCore;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import static org.junit.jupiter.api.Assertions.assertEquals;


@TestInstance(TestInstance.Lifecycle.PER_METHOD)
public class CudaTensorTest {

    ISciCore sciCore;

    @BeforeEach
    void setUp() {
        sciCore = new SciCore();
        sciCore.setBackend(SciCore.BackendType.CUDA);
    }


    @Test
    void testCreateByteArray() {
        byte[] array = new byte[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        ITensor tensor = sciCore.array(array);
        for (int i = 0; i < array.length; i++) {
            assertEquals(array[i], tensor.getByte(i));
        }
    }

    @Test
    void testCreateShortArray() {
        short[] array = new short[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        ITensor tensor = sciCore.array(array);
        for (int i = 0; i < array.length; i++) {
            assertEquals(array[i], tensor.getShort(i));
        }
    }

    @Test
    void testCreateIntArray() {
        int[] array = new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        ITensor tensor = sciCore.array(array);
        for (int i = 0; i < array.length; i++) {
            assertEquals(array[i], tensor.getInt(i));
        }
    }

    @Test
    void testCreateLongArray() {
        long[] array = new long[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        ITensor tensor = sciCore.array(array);
        for (int i = 0; i < array.length; i++) {
            assertEquals(array[i], tensor.getLong(i));
        }
    }

    @Test
    void testCreateFloatArray() {
        float[] array = new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        ITensor tensor = sciCore.array(array);
        for (int i = 0; i < array.length; i++) {
            assertEquals(array[i], tensor.getFloat(i));
        }
    }

    @Test
    void testCreateDoubleArray() {
        double[] array = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        ITensor tensor = sciCore.array(array);
        for (int i = 0; i < array.length; i++) {
            assertEquals(array[i], tensor.getDouble(i));
        }
    }

    @Test
    void testCreateBooleanArray() {
        boolean[] array = new boolean[]{true, false, true, false, true, false, true, false, false, false};
        ITensor tensor = sciCore.array(array);
        for (int i = 0; i < array.length; i++) {
            assertEquals(array[i], tensor.getBoolean(i));
        }
    }

}
