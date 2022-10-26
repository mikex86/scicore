package me.mikex86.scicore.utils;

import me.mikex86.scicore.ITensor;
import org.jetbrains.annotations.NotNull;

public class GradientUtil {

    @NotNull
    public static ITensor sumGradientsOnBroadDims(@NotNull ITensor tmpGradients, long[] shapeOfParameter) {
        long[] gradientShape = tmpGradients.getShape();
        for (int i = 0; i < gradientShape.length; i++) {
            if (shapeOfParameter.length - i - 1 >= 0) {
                if (gradientShape[gradientShape.length - i - 1] != shapeOfParameter[shapeOfParameter.length - i - 1]) {
                    tmpGradients = tmpGradients.reduceSum(i, true);
                }
            } else {
                tmpGradients = tmpGradients.reduceSum(0, false);
            }
        }
        return tmpGradients;
    }

}
