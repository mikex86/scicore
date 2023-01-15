package me.mikex86.scicore.backend.impl.cuda.op;

import jcuda.Pointer;
import me.mikex86.scicore.backend.impl.cuda.CudaBackend;
import me.mikex86.scicore.backend.impl.cuda.CudaTensor;
import me.mikex86.scicore.backend.impl.cuda.codegen.BroadcastingElementWiseOperationKernelCodeGenerator;
import me.mikex86.scicore.backend.impl.cuda.codegen.DataTypeUtils;
import me.mikex86.scicore.backend.impl.cuda.codegen.KernelCodeGenerator;
import me.mikex86.scicore.backend.impl.cuda.kernel.CudaKernel;
import me.mikex86.scicore.backend.impl.cuda.kernel.CudaKernelLaunchConfig;
import me.mikex86.scicore.backend.impl.cuda.memory.CudaMemoryHandle;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.utils.GradientUtil;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public class CudaPowOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final CudaBackend backend;

    public CudaPowOp(@NotNull CudaBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shapeA = a.getShape();
        long[] shapeB = b.getShape();

        long[] stridesA = a.getStrides();
        long[] stridesB = b.getStrides();

        long[] resultShape = ShapeUtils.broadcastShapes(shapeA, shapeB);

        DataType dataTypeA = a.getDataType();
        DataType dataTypeB = b.getDataType();
        DataType resultDataType = DataType.getLarger(dataTypeA, dataTypeB);

        long nElements = ShapeUtils.getNumElements(resultShape);

        CudaTensor result = this.backend.createTensor(resultDataType, resultShape);

        long[] resultStrides = result.getStrides();

        CudaMemoryHandle aDevicePtr = backend.getCudaMemoryManager().ensureOnDevice(a);
        CudaMemoryHandle bDevicePtr = backend.getCudaMemoryManager().ensureOnDevice(b);
        int threadsPerBlock = 1024;
        int nBlocks = Math.toIntExact((nElements + threadsPerBlock - 1) / threadsPerBlock);

        CudaKernel kernel = CudaKernel.jitCompile(
                KernelCodeGenerator
                        .create()
                        .addFunction(
                                KernelCodeGenerator.KernelFunction
                                        .builder()
                                        .prefix("extern \"C\" __global__")
                                        .returnType("void")
                                        .functionName("pow_kernel")
                                        .parameter(dataTypeA, 1, "a")
                                        .parameter(dataTypeB, 1, "b")
                                        .parameter(resultDataType, 1, "out")
                                        .body(
                                                BroadcastingElementWiseOperationKernelCodeGenerator
                                                        .builder()
                                                        .shapeA(shapeA)
                                                        .shapeB(shapeB)
                                                        .resultShape(resultShape)
                                                        .stridesA(stridesA)
                                                        .stridesB(stridesB)
                                                        .resultStrides(resultStrides)
                                                        .operation("pow")
                                                        .build()
                                                        .generateCode()
                                        )
                                        .build()
                        )
                        .buildCode(), List.of("pow_kernel"));
        kernel.launchBlocking(
                "pow_kernel",
                CudaKernelLaunchConfig.builder()
                        .blockDimX(threadsPerBlock)
                        .gridDimX(nBlocks)
                        .parameters(
                                Pointer.to(
                                        Pointer.to(aDevicePtr.getDevicePointer()),
                                        Pointer.to(bDevicePtr.getDevicePointer()),
                                        Pointer.to(result.getDataContainer().getDeviceMemoryHandle().getDevicePointer())
                                )
                        )
                        .build()
        );

        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shapeA = a.getShape();
        long[] shapeB = b.getShape();
        long[] finalShape = ShapeUtils.broadcastShapes(shapeA, shapeB);

        DataType dataTypeA = a.getDataType();
        DataType dataTypeB = b.getDataType();

        DataType resultDataType = DataType.getLarger(dataTypeA, dataTypeB);

        return new LazyTensor(backend, finalShape, resultDataType);
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
                        try (ITensor globalGradients = upstreamGradient.multiply(localGradients)) { // dL/dA = dL/dP * dP/dA
                            a.accumulateGradient(globalGradients);
                        }
                    }
                }
            }
        }

        if (b.requiresGradients()) {
            try (ITensor localGradients = backend.createTensor(resultDataType, shapeA)) {
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
                try (ITensor globalGradients = upstreamGradient.matmul(localGradients, false, true)) { // multiply and sum = matmul in this case
                    b.accumulateGradient(globalGradients);
                }
            }
        }
    }
}
