package me.mikex86.scicore.backend.impl.genericcpu.jni;

import me.mikex86.scicore.tensor.DataType;
import org.jetbrains.annotations.NotNull;

import static me.mikex86.scicore.backend.impl.genericcpu.jni.JNIDataTypes.dataTypeToInt;

public class SigmoidJNI {

    private static native void nsigmoid(long inPtr, long outPtr, long nElements, int dataType);

    public static void sigmoid(long inPtr, long outPtr, long nElements, @NotNull DataType dataType) {
        nsigmoid(inPtr, outPtr, nElements, dataTypeToInt(dataType));
    }

}
