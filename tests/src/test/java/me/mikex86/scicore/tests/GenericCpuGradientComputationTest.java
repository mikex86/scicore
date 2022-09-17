package me.mikex86.scicore.tests;

import me.mikex86.scicore.ISciCore;

public class GenericCpuGradientComputationTest extends GradientComputationTest {

    GenericCpuGradientComputationTest() {
        super(ISciCore.BackendType.GENERIC_CPU);
    }
}
