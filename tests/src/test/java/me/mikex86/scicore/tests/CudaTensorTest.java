package me.mikex86.scicore.tests;

import me.mikex86.scicore.ISciCore;
import org.junit.jupiter.api.condition.EnabledIf;

@EnabledIf("me.mikex86.scicore.tests.CudaTensorTest#isCudaAvailable")
public class CudaTensorTest extends TensorTest {

    CudaTensorTest() {
        super(ISciCore.BackendType.CUDA);
    }

    static boolean isCudaAvailable() {
        return ISciCore.isBackendAvailable(ISciCore.BackendType.CUDA);
    }

}
