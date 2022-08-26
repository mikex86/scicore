package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.backend.impl.jvm.JvmTensor;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.op.Graph;
import me.mikex86.scicore.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

import java.awt.*;

public class JvmPlusOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final ISciCoreBackend backend;

    public JvmPlusOp(@NotNull ISciCoreBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shapeA = a.getShape();
        long[] shapeB = b.getShape();

        long aNumElements = ShapeUtils.getNumElements(shapeA);
        long bNumElements = ShapeUtils.getNumElements(shapeB);

        long[] finalShape = shapeA;
        boolean broadcast = false;
        if (!ShapeUtils.equals(shapeA, shapeB)) {
            finalShape = ShapeUtils.broadcastShapes(shapeA, shapeB);
            broadcast = true;
        }

        DataType dataTypeA = a.getDataType();
        DataType dataTypeB = b.getDataType();
        DataType resultDataType = DataType.getLarger(dataTypeA, dataTypeB);

        long nElements = ShapeUtils.getNumElements(finalShape);

        ITensor tensor = new JvmTensor(this.backend, resultDataType, finalShape);
        if (broadcast) {
            for (long i = 0; i < nElements; i++) {
                if (resultDataType.isFloatingPoint()) {
                    double aV = a.getAsDoubleFlat(i % aNumElements);
                    double bV = b.getAsDoubleFlat(i % bNumElements);
                    double resultVal = aV + bV;
                    tensor.setByDoubleFlat(resultVal, i);
                } else {
                    long aV = a.getAsLongFlat(i % aNumElements);
                    long bV = b.getAsLongFlat(i % bNumElements);
                    long resultVal = aV + bV;
                    tensor.setByLongFlat(resultVal, i);
                }
            }
        } else {
            for (long i = 0; i < nElements; i++) {
                if (resultDataType.isFloatingPoint()) {
                    double aV = a.getAsDoubleFlat(i);
                    double bV = b.getAsDoubleFlat(i);
                    double resultVal = aV + bV;
                    tensor.setByDoubleFlat(resultVal, i);
                }
            }
        }
        return tensor;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shapeA = a.getShape();
        long[] finalShape = shapeA;
        long[] shapeB = b.getShape();
        if (!ShapeUtils.equals(shapeA, shapeB)) {
            finalShape = ShapeUtils.broadcastShapes(shapeA, shapeB);
        }
        DataType dataTypeA = a.getDataType();
        DataType dataTypeB = b.getDataType();
        DataType resultDataType = DataType.getLarger(dataTypeA, dataTypeB);
        return new LazyTensor(backend, finalShape, resultDataType, () -> perform(ctx, a, b));
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient a, @NotNull IGraph.ITensorNodeWithGradient b) {
        // TODO: DO THIS FOR MINUS AS WELL

        // Note that the upstream gradient dL/dz where z is the output of the current node
        // is with respect to all parameters a(p11,p12,...p1n) and b(p21,p22,...p2n) where are and b are the
        // inputs to the current node.
        // When computing the gradient for a, we need to sum over all the other parameters in b, and vice versa.

        if (a.requiresGradients()) {
            ITensor aValue = a.getValue();

            long[] shapeA = aValue.getShape();

            ITensor gradients = backend.createTensor(upstreamGradient.getDataType(), shapeA);
            gradients.fill(1);
            gradients = gradients.multiply(upstreamGradient);

            long[] gradientShape = gradients.getShape();

            if (ShapeUtils.compareBroadcastRank(gradientShape, shapeA) > 0) {
                int nCommonDimensions = ShapeUtils.getNumNotCommonDimensions(shapeA, gradientShape);
                for (int i = 0; i < nCommonDimensions; i++) {
                    gradients = gradients.reduceSum(0);
                }
            }

            a.accumulateGradient(gradients);
        }

        if (b.requiresGradients()) {
            ITensor bValue = b.getValue();

            long[] shapeB = bValue.getShape();

            ITensor gradients = backend.createTensor(upstreamGradient.getDataType(), shapeB);
            gradients.fill(1);
            gradients = gradients.multiply(upstreamGradient);

            long[] gradientsShape = gradients.getShape();

            if (ShapeUtils.compareBroadcastRank(gradientsShape, shapeB) > 0) {
                int nCommonDimensions = ShapeUtils.getNumNotCommonDimensions(gradientsShape, shapeB);
                for (int i = 0; i < nCommonDimensions; i++) {
                    gradients = gradients.reduceSum(0);
                }
            }

            b.accumulateGradient(gradients);
        }
    }
}
