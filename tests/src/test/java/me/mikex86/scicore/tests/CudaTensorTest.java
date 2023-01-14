package me.mikex86.scicore.tests;

import me.mikex86.scicore.ISciCore;

public class CudaTensorTest extends TensorTest {

    CudaTensorTest() {
        super(ISciCore.BackendType.CUDA);
    }

}
