package me.mikex86.scicore.tests;

import me.mikex86.scicore.ISciCore;

public class GenCPUApproxLinearFuncTrainingTest extends ApproxLinearFuncTrainingTest{

    protected GenCPUApproxLinearFuncTrainingTest() {
        super(ISciCore.BackendType.CPU);
    }

}
