package me.mikex86.scicore.backend.impl.genericcpu.jni;

import me.mikex86.scicore.tensor.DataType;
import org.jetbrains.annotations.NotNull;

import static me.mikex86.scicore.backend.impl.genericcpu.jni.JNIDataTypes.dataTypeToInt;

public class ExpJNI {

    private static native void nexp(long inPtr, long outPtr, long nElements, int dataType);

    public static void exp(long inPtr, long outPtr, long nElements, @NotNull DataType dataType) {
        nexp(inPtr, outPtr, nElements, dataTypeToInt(dataType));
    }

}
