package me.mikex86.scicore.backend.impl.genericcpu.jni;

import me.mikex86.scicore.tensor.DataType;
import org.jetbrains.annotations.NotNull;

import static me.mikex86.scicore.backend.impl.genericcpu.jni.JNIDataTypes.dataTypeToInt;

public class ReluJNI {

    private static native void nrelu(long inPtr, long outPtr, long nElements, int dataType);

    public static void relu(long inPtr, long outPtr, long nElements, @NotNull DataType dataType) {
        nrelu(inPtr, outPtr, nElements, dataTypeToInt(dataType));
    }

    private static native void nreluGradients(long inPtr, long outPtr, long nElements, int dataType);

    public static void reluGradients(long inPtr, long outPtr, long nElements, @NotNull DataType dataType) {
        nreluGradients(inPtr, outPtr, nElements, dataTypeToInt(dataType));
    }

}
