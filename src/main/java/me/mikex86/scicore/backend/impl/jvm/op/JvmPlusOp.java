package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.backend.impl.jvm.JvmTensor;
import me.mikex86.scicore.backend.impl.jvm.JvmDerivedTensor;
import me.mikex86.scicore.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

public class JvmPlusOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final ISciCoreBackend backend;

    public JvmPlusOp(@NotNull ISciCoreBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull ITensor a, @NotNull ITensor b) {
        // TODO: BROADCASTING
        long[] shapeA = a.getShape();
        long[] shapeB = b.getShape();
        Validator.assertTrue(ShapeUtils.equals(shapeA, shapeB), "Shapes must match during addition for now... TODO: Broadcasting");

        long nElements = ShapeUtils.getNumElements(shapeA);

        DataType dataType = DataType.getLarger(a.getDataType(), b.getDataType());
        ITensor tensor = new JvmTensor(this.backend, dataType, shapeA);

        for (long i = 0; i < nElements; i++) {
            if (dataType.isFloatingPoint()) {
                double aV = a.getAsDoubleFlat(i);
                double bV = b.getAsDoubleFlat(i);
                double resultVal = aV + bV;
                tensor.setByDoubleFlat(resultVal, i);
            }
        }
        return tensor;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull ITensor a, @NotNull ITensor b) {
        long[] shapeA = a.getShape();
        long[] shapeB = b.getShape();
        long[] outputShape = ShapeUtils.broadcastShapes(shapeA, shapeB);
        DataType ownDataType = a.getDataType();
        DataType otherDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(ownDataType, otherDataType);
        return new JvmDerivedTensor(backend, outputShape, resultDataType, () -> perform(a, b));
    }

    @Override
    public void computeGradients(@NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient a, @NotNull IGraph.ITensorNodeWithGradient b) {
        a.accumulateGradient(upstreamGradient);
        b.accumulateGradient(upstreamGradient);
    }
}
