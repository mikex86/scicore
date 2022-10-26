package me.mikex86.scicore.backend.impl.genericcpu.jni;

import me.mikex86.scicore.DataType;

import static me.mikex86.scicore.backend.impl.genericcpu.jni.JNIDataTypes.dataTypeToInt;

public class ArgmaxJNI {

    private static native void nargmax(long aPtr, long[] shapeA, long[] stridesA,
                                       long cPtr, long[] shapeC, long[] stridesC,
                                       int dataType,
                                       long dim);


    public static void argmax(long aPtr, long[] shapeA, long[] stridesA,
                              long cPtr, long[] shapeC, long[] stridesC,
                              DataType dataType,
                              long dim) {
        nargmax(
                aPtr, shapeA, stridesA,
                cPtr, shapeC, stridesC,
                dataTypeToInt(dataType),
                dim
        );
    }

}