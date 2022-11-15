package me.mikex86.scicore.backend.impl.genericcpu.jni;

import me.mikex86.scicore.tensor.DataType;
import org.jetbrains.annotations.NotNull;

import static me.mikex86.scicore.backend.impl.genericcpu.jni.JNIDataTypes.dataTypeToInt;

public class TanhJNI {

    private static native void ntanh(long inPtr, long outPtr, long nElements, int dataType);

    public static void tanh(long inPtr, long outPtr, long nElements, @NotNull DataType dataType) {
        ntanh(inPtr, outPtr, nElements, dataTypeToInt(dataType));
    }

}
