package me.mikex86.scicore.backend.impl.genericcpu.jni;

import me.mikex86.scicore.tensor.DataType;

import static me.mikex86.scicore.backend.impl.genericcpu.jni.JNIDataTypes.dataTypeToInt;

public class ReduceSumJNI {

    private static native void nreduceSum(long aPtr,
                                          long cPtr,
                                          int dataType,
                                          long[] shapeA, long[] stridesA,
                                          long[] shapeC, long[] stridesC,
                                          long dimension, boolean keepDimensions);

    public static void reduceSum(long aPtr,
                                 long cPtr,
                                 DataType dataType,
                                 long[] shapeA, long[] stridesA,
                                 long[] shapeC, long[] stridesC,
                                 long dimension, boolean keepDimensions) {
        nreduceSum(aPtr, cPtr, dataTypeToInt(dataType), shapeA, stridesA, shapeC, stridesC, dimension, keepDimensions);
    }

}
