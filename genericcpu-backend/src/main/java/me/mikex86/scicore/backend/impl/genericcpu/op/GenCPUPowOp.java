package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.backend.impl.genericcpu.jni.PowJNI;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

public class GenCPUPowOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final GenCPUBackend backend;

    public GenCPUPowOp(@NotNull GenCPUBackend backend) {
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
        DirectMemoryHandle aHandle = a.getContentsAsDirectMemory();
        DirectMemoryHandle bHandle = b.getContentsAsDirectMemory();
        DirectMemoryHandle resultMemory = result.getContentsAsDirectMemory();

        PowJNI.pow(aHandle.getNativePtr(), a.getShape(), a.getStrides(), dataTypeA, bHandle.getNativePtr(), b.getShape(), b.getStrides(), dataTypeB, resultMemory.getNativePtr(), result.getShape(), result.getStrides(), resultDataType);

        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        Validator.assertTrue(b.isScalar(), "Exponent must be scalar"); // Only supporting scalar exponents for now
        long[] shapeA = a.getShape();
        DataType dataTypeA = a.getDataType();
        DataType dataTypeB = b.getDataType();
        DataType resultDataType = DataType.getLarger(dataTypeA, dataTypeB);
        return new LazyTensor(backend, shapeA, resultDataType);
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient a, @NotNull IGraph.ITensorNodeWithGradient b) {
        ITensor bValue = b.getValue();
        Validator.assertTrue(bValue.isScalar(), "Exponent must be scalar"); // Only supporting scalar exponents for now
        ITensor aValue = a.getValue();
        long[] shapeA = aValue.getShape();
        DataType dataTypeA = aValue.getDataType();
        DataType dataTypeB = bValue.getDataType();
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
            // dP/dA = B * A ^ (B - 1)
            try (ITensor bMinusOne = bValue.minus(1f)) {
                try (ITensor aPowBMinusOne = aValue.pow(bMinusOne)) {
                    try (ITensor localGradients = bValue.multiply(aPowBMinusOne)) {
                        ITensor globalGradients = upstreamGradient.multiply(localGradients); // dL/dA = dL/dP * dP/dA
                        a.accumulateGradient(globalGradients);
                    }
                }
            }
        }

        if (b.requiresGradients()) {
            ITensor localGradients = backend.createTensor(resultDataType, shapeA);
            // TODO: OPTIMIZE
            for (long i = 0; i < nElements; i++) {
                if (resultDataType.isFloatingPoint()) {
                    double exponent = bValue.elementAsDouble();
                    double aV = aValue.getAsDoubleFlat(i);
                    double resultVal = Math.pow(aV, exponent);
                    localGradients.setByDoubleFlat(resultVal * Math.log(aV), i); // dP/dB = A ^ B * ln(A)
                } else {
                    long exponent = bValue.elementAsLong();
                    long aV = aValue.getAsLongFlat(i);
                    long resultVal = (long) Math.pow(aV, exponent);
                    localGradients.setByLongFlat((long) (resultVal * Math.log(aV)), i); // dP/dB = A ^ B * ln(A)
                }
            }
            // dL/dB = dL/dP * dP/dB
            ITensor globalGradients = upstreamGradient.matmul(localGradients, false, true); // multiply and sum = matmul in this case
            b.accumulateGradient(globalGradients);
        }
    }
}
