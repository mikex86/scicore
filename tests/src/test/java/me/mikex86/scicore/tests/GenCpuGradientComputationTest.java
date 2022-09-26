package me.mikex86.scicore.tests;

import me.mikex86.scicore.ISciCore;

public class GenCpuGradientComputationTest extends GradientComputationTest {

    GenCpuGradientComputationTest() {
        super(ISciCore.BackendType.GENERIC_CPU);
    }
}
