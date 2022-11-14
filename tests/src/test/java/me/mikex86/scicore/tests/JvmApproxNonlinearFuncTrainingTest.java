package me.mikex86.scicore.tests;

import me.mikex86.scicore.ISciCore;
import org.jetbrains.annotations.NotNull;

public class JvmApproxNonlinearFuncTrainingTest extends ApproxNonlinearFuncTrainingTest {

    protected JvmApproxNonlinearFuncTrainingTest() {
        super(ISciCore.BackendType.JVM);
    }

}
