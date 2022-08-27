package me.mikex86.scicore.backend.impl.cuda;

import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.SciCore;

public class Test {

    public static void main(String[] args) {
        SciCore sciCore = new SciCore();
        sciCore.setBackend(ISciCore.BackendType.CUDA);
    }

}
