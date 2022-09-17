package me.mikex86.scicore.tests;

import me.mikex86.scicore.ISciCore;

public class JvmGradientComputationTest extends GradientComputationTest {

    JvmGradientComputationTest() {
        super(ISciCore.BackendType.JVM);
    }
}
