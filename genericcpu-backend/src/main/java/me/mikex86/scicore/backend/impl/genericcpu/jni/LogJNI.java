package me.mikex86.scicore.backend.impl.genericcpu.jni;

import me.mikex86.scicore.tensor.DataType;
import org.jetbrains.annotations.NotNull;

import static me.mikex86.scicore.backend.impl.genericcpu.jni.JNIDataTypes.dataTypeToInt;

public class LogJNI {

    private static native void nlog(long inPtr, long outPtr, long nElements, int dataType);

    public static void log(long inPtr, long outPtr, long nElements, @NotNull DataType dataType) {
        nlog(inPtr, outPtr, nElements, dataTypeToInt(dataType));
    }

}
