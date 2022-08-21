package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.op.Graph;
import me.mikex86.scicore.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

public class JvmPowerOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final ISciCoreBackend backend;

    public JvmPowerOp(@NotNull ISciCoreBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        Validator.assertTrue(b.isScalar(), "Exponent must be scalar"); // Only supporting scalar exponents for now
        long[] shapeA = a.getShape();
        DataType dataTypeA = a.getDataType();
        DataType dataTypeB = b.getDataType();
        DataType resultDataType = DataType.getLarger(dataTypeA, dataTypeB);

        ITensor result = backend.createTensor(resultDataType, shapeA);
        long nElements = ShapeUtils.getNumElements(shapeA);
        for (long i = 0; i < nElements; i++) {
            if (resultDataType.isFloatingPoint()) {
                double exponent = b.elementAsDouble();
                double aV = a.getAsDoubleFlat(i);
                double resultVal = Math.pow(aV, exponent);
                result.setByDoubleFlat(resultVal, i);
            } else {
                long exponent = b.elementAsLong();
                long aV = a.getAsLongFlat(i);
                long resultVal = (long) Math.pow(aV, exponent);
                result.setByLongFlat(resultVal, i);
            }
        }
        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        Validator.assertTrue(b.isScalar(), "Exponent must be scalar"); // Only supporting scalar exponents for now
        long[] shapeA = a.getShape();
        DataType dataTypeA = a.getDataType();
        DataType dataTypeB = b.getDataType();
        DataType resultDataType = DataType.getLarger(dataTypeA, dataTypeB);
        return new LazyTensor(backend, shapeA, resultDataType, () -> perform(ctx, a, b));
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient a, @NotNull IGraph.ITensorNodeWithGradient b) {
        Validator.assertTrue(b.getValue().isScalar(), "Exponent must be scalar"); // Only supporting scalar exponents for now
        long[] shapeA = a.getValue().getShape();
        DataType dataTypeA = a.getValue().getDataType();
        DataType dataTypeB = b.getValue().getDataType();
        DataType resultDataType = DataType.getLarger(dataTypeA, dataTypeB);
        long nElements = ShapeUtils.getNumElements(shapeA);

        // Notation:
        // A = a (base)
        // B = b (exponent)
        // P = A ^ B
        // dL/dP = upstreamGradient for chain rule

        // Gradients:
        // dL/dA = dL/dP * dP/dA  // chain rule
        // dL/dA = B * A ^ (B - 1) // this is the normal power rule

        // dL/dB = dL/dP * dP/dB  // chain rule
        // dL/dB = A ^ B * ln(A)  // this is the exponentiation rule

        if (a.requiresGradients()) {
            ITensor localGradients = backend.createTensor(resultDataType, shapeA);
            for (long i = 0; i < nElements; i++) {
                if (resultDataType.isFloatingPoint()) {
                    double exponent = b.getValue().elementAsDouble();
                    double aV = a.getValue().getAsDoubleFlat(i);
                    localGradients.setByDoubleFlat(exponent * Math.pow(aV, exponent - 1), i); // dP/dA = B * A ^ (B - 1)
                } else {
                    long exponent = b.getValue().elementAsLong();
                    long aV = a.getValue().getAsLongFlat(i);
                    localGradients.setByLongFlat((long) (exponent * Math.pow(aV, exponent - 1)), i); // dP/dA = B * A ^ (B - 1)
                }
            }
            ITensor globalGradients = upstreamGradient.multiply(localGradients); // dL/dA = dL/dP * dP/dA
            a.accumulateGradient(globalGradients);
        }

        if (b.requiresGradients()) {
            ITensor localGradients = backend.createTensor(resultDataType, shapeA);
            for (long i = 0; i < nElements; i++) {
                if (resultDataType.isFloatingPoint()) {
                    double exponent = b.getValue().elementAsDouble();
                    double aV = a.getValue().getAsDoubleFlat(i);
                    double resultVal = Math.pow(aV, exponent);
                    localGradients.setByDoubleFlat(resultVal * Math.log(aV), i); // dP/dB = A ^ B * ln(A)
                } else {
                    long exponent = b.getValue().elementAsLong();
                    long aV = a.getValue().getAsLongFlat(i);
                    long resultVal = (long) Math.pow(aV, exponent);
                    localGradients.setByLongFlat((long) (resultVal * Math.log(aV)), i); // dP/dB = A ^ B * ln(A)
                }
            }
            ITensor globalGradients = upstreamGradient.matmul(localGradients.transpose()); // dL/dB = dL/dP * dP/dB
            b.accumulateGradient(globalGradients);
        }
    }
}
