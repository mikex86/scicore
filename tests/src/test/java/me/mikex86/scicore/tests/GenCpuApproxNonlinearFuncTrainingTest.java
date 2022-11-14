package me.mikex86.scicore.tests;

import me.mikex86.scicore.ISciCore;

public class GenCpuApproxNonlinearFuncTrainingTest extends ApproxNonlinearFuncTrainingTest {

    protected GenCpuApproxNonlinearFuncTrainingTest() {
        super(ISciCore.BackendType.CPU);
    }

}
