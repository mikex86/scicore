package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.backend.impl.jvm.JvmTensor;
import me.mikex86.scicore.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

public class JvmMultiplyOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final ISciCoreBackend backend;

    public JvmMultiplyOp(@NotNull ISciCoreBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull ITensor a, @NotNull ITensor b) {
        long[] shapeA = a.getShape();
        long[] shapeB = b.getShape();
        long aNumElements = ShapeUtils.getNumElements(shapeA);
        long bNumElements = ShapeUtils.getNumElements(shapeB);

        long[] finalShape = shapeA;
        boolean broadcast = false;
        if (!b.isScalar()) {
            if (!ShapeUtils.equals(shapeA, shapeB)) {
                finalShape = ShapeUtils.broadcastShapes(shapeA, shapeB);
                broadcast = true;
            }
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
                    double resultVal = aV * bV;
                    tensor.setByDoubleFlat(resultVal, i);
                } else {
                    long aV = a.getAsLongFlat(i % aNumElements);
                    long bV = b.getAsLongFlat(i % bNumElements);
                    long resultVal = aV * bV;
                    tensor.setByLongFlat(resultVal, i);
                }
            }
        } else {
            for (long i = 0; i < nElements; i++) {
                if (resultDataType.isFloatingPoint()) {
                    double aV = a.getAsDoubleFlat(i);
                    double bV = b.isScalar() ? b.elementAsDouble() : b.getAsDoubleFlat(i);
                    double resultVal = aV * bV;
                    tensor.setByDoubleFlat(resultVal, i);
                } else {
                    long aV = a.getAsLongFlat(i);
                    long bV = b.isScalar() ? b.elementAsLong() : b.getAsLongFlat(i);
                    long resultVal = aV * bV;
                    tensor.setByLongFlat(resultVal, i);
                }
            }
        }
        return tensor;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull ITensor a, @NotNull ITensor b) {
        long[] shapeA = a.getShape();
        long[] finalShape = shapeA;
        if (!b.isScalar()) {
            long[] shapeB = b.getShape();
            if (!ShapeUtils.equals(shapeA, shapeB)) {
                finalShape = ShapeUtils.broadcastShapes(shapeA, shapeB);
            }
        }
        DataType dataTypeA = a.getDataType();
        DataType dataTypeB = b.getDataType();
        DataType resultDataType = DataType.getLarger(dataTypeA, dataTypeB);
        return new LazyTensor(backend, finalShape, resultDataType, () -> perform(a, b));
    }

    @Override
    public void computeGradients(@NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient a, @NotNull IGraph.ITensorNodeWithGradient b) {
        a.accumulateGradient(upstreamGradient.multiply(b.getValue()));
        b.accumulateGradient(upstreamGradient.multiply(a.getValue()));
    }
}
