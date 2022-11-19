package me.mikex86.scicore.utils;

import me.mikex86.scicore.profiling.Profiler;
import me.mikex86.scicore.tensor.ITensor;
import org.jetbrains.annotations.NotNull;

public class GradientUtil {

    /**
     * This method is used to sum over dimension, which were broadcast in the forward pass.
     * We need to sum over these dimensions, because all these gradients stem from the same scalar in the original parameter, which was virtually expanded.
     * @param tmpGradients dL/dZ where Z is the output of the forward pass
     * @param shapeOfParameter The shape of the parameter, which was broadcast.
     * @return dL/dP where P is the parameter, which was broadcast.
     */
    @NotNull
    public static ITensor sumGradientsOnBroadcastDims(@NotNull ITensor tmpGradients, long[] shapeOfParameter) {
        long[] gradientShape = tmpGradients.getShape();
        for (int i = 0; i < gradientShape.length; i++) {
            if (shapeOfParameter.length - i - 1 >= 0) {
                if (gradientShape[gradientShape.length - i - 1] != shapeOfParameter[shapeOfParameter.length - i - 1]) {
                    tmpGradients = tmpGradients.reduceSum(gradientShape.length - i - 1, true);
                }
            } else {
                tmpGradients = tmpGradients.reduceSum(0, false);
            }
        }
        return tmpGradients;
    }

}
