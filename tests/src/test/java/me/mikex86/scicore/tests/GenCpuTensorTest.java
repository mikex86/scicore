package me.mikex86.scicore.tests;

import me.mikex86.scicore.ISciCore;

public class GenCpuTensorTest extends TensorTest {

    GenCpuTensorTest() {
        super(ISciCore.BackendType.CPU);
    }

}
