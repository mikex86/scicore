package me.mikex86.scicore.tests;

import me.mikex86.scicore.ISciCore;

public class CpuTensorTest extends TensorTest {

    CpuTensorTest() {
        super(ISciCore.BackendType.GENERIC_CPU);
    }

}
