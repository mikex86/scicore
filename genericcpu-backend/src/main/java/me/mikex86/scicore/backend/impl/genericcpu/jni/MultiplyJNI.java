package me.mikex86.scicore.backend.impl.genericcpu.jni;

import me.mikex86.scicore.tensor.DataType;

import static me.mikex86.scicore.backend.impl.genericcpu.jni.JNIDataTypes.dataTypeToInt;

public class MultiplyJNI {

    private static native void nmultiply(long aPtr, long[] shapeA, long[] stridesA, int dataTypeA,
                                         long bPtr, long[] shapeB, long[] stridesB, int dataTypeB,
                                         long cPtr, long[] shapeC, long[] stridesC, int dataTypeC);

    public static void multiply(long aPtr, long[] shapeA, long[] stridesA, DataType dataTypeA,
                                long bPtr, long[] shapeB, long[] stridesB, DataType dataTypeB,
                                long cPtr, long[] shapeC, long[] stridesC, DataType dataTypeC) {
        nmultiply(
                aPtr, shapeA, stridesA, dataTypeToInt(dataTypeA),
                bPtr, shapeB, stridesB, dataTypeToInt(dataTypeB),
                cPtr, shapeC, stridesC, dataTypeToInt(dataTypeC)
        );
    }

}
