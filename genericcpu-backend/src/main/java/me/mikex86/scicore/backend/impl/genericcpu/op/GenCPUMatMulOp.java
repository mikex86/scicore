package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.backend.impl.genericcpu.GenCPUTensor;
import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.backend.impl.genericcpu.jni.MatmulJNI;
import me.mikex86.scicore.op.Graph;
import me.mikex86.scicore.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.utils.Pair;
import me.mikex86.scicore.utils.ShapeUtils;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

import java.nio.*;

import static me.mikex86.scicore.backend.impl.genericcpu.jni.MatmulJNI.*;

public class GenCPUMatMulOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final GenCPUBackend backend;

    public GenCPUMatMulOp(@NotNull GenCPUBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shape = a.getShape();
        long[] otherShape = b.getShape();
        Validator.assertTrue(otherShape.length == 2, "Only 2D matrices are supported");
        Validator.assertTrue(shape.length == 2, "Only 2D matrices are supported");
        Validator.assertTrue(shape[1] == otherShape[0], "Shape mismatch. A.shape[1] != B.shape[0]");
        Validator.assertTrue(a.getDataType().isNumeric(), "Data type of A is not numeric");
        Validator.assertTrue(b.getDataType().isNumeric(), "Data type of B is not numeric");
        long[] resultShape = ShapeUtils.matrixMultiplyShape(shape, otherShape);
        DataType ownDataType = a.getDataType();
        DataType otherDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(ownDataType, otherDataType);

        GenCPUTensor result = new GenCPUTensor(this.backend, resultDataType, resultShape);

        long m = shape[0], n = otherShape[1], k = shape[1];

        try (MemoryStack stack = MemoryStack.stackPush()) {
            Buffer factor;
            switch (ownDataType) {
                case INT8: {
                    ByteBuffer ptr = stack.malloc(1);
                    ptr.put((byte) 1);
                    ptr.flip();
                    factor = ptr;
                    break;
                }
                case INT16: {
                    ShortBuffer ptr = stack.mallocShort(1);
                    ptr.put((short) 1);
                    ptr.flip();
                    factor = ptr;
                    break;
                }
                case INT32: {
                    IntBuffer ptr = stack.mallocInt(1);
                    ptr.put(1);
                    ptr.flip();
                    factor = ptr;
                    break;
                }
                case INT64: {
                    LongBuffer ptr = stack.mallocLong(1);
                    ptr.put(1L);
                    ptr.flip();
                    factor = ptr;
                    break;
                }
                case FLOAT32: {
                    FloatBuffer ptr = stack.mallocFloat(1);
                    ptr.put(1.0f);
                    ptr.flip();
                    factor = ptr;
                    break;
                }
                case FLOAT64: {
                    DoubleBuffer ptr = stack.mallocDouble(1);
                    ptr.put(1.0);
                    ptr.flip();
                    factor = ptr;
                    break;
                }
                default:
                    throw new IllegalArgumentException("Unsupported data type: " + ownDataType);
            }

            long aPtr, bPtr;
            boolean isACopy = false, isBCopy = false;
            {
                GenCPUTensor aTensor = a.getIfIsType(GenCPUTensor.class);
                GenCPUTensor bTensor = b.getIfIsType(GenCPUTensor.class);
                // handle a tensor
                {
                    if (aTensor != null) {
                        aPtr = aTensor.getDataContainer().getDataPtr();
                    } else {
                        Pair<ByteBuffer, Boolean> pair = a.getAsDirectBuffer();
                        aPtr = MemoryUtil.memAddress(pair.getFirst());
                        isACopy = pair.getSecond();
                    }
                }
                // handle b tensor
                {
                    if (bTensor != null) {
                        bPtr = bTensor.getDataContainer().getDataPtr();
                    } else {
                        Pair<ByteBuffer, Boolean> pair = b.getAsDirectBuffer();
                        bPtr = MemoryUtil.memAddress(pair.getFirst());
                        isBCopy = pair.getSecond();
                    }
                }
            }

            matmul(OP_NONE, OP_NONE,
                    m, n, k,
                    MemoryUtil.memAddress(factor),
                    aPtr,
                    getMatmulDataType(ownDataType),
                    m,
                    MemoryUtil.memAddress(factor),
                    bPtr,
                    getMatmulDataType(otherDataType),
                    k,
                    result.getDataContainer().getDataPtr(),
                    getMatmulDataType(resultDataType),
                    m
            );

            if (isACopy) {
                backend.getMemoryManager().free(aPtr);
            }
            if (isBCopy) {
                backend.getMemoryManager().free(bPtr);
            }
        }
        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shape = a.getShape();
        long[] otherShape = b.getShape();
        Validator.assertTrue(otherShape.length == 2, "Only 2D matrices are supported");
        Validator.assertTrue(shape.length == 2, "Only 2D matrices are supported");
        Validator.assertTrue(shape[1] == otherShape[0], "Shape mismatch. A.shape[1] != B.shape[0]");
        Validator.assertTrue(a.getDataType().isNumeric(), "Data type of A is not numeric");
        Validator.assertTrue(b.getDataType().isNumeric(), "Data type of B is not numeric");
        long[] resultShape = ShapeUtils.matrixMultiplyShape(shape, otherShape);
        DataType ownDataType = a.getDataType();
        DataType otherDataType = b.getDataType();
        // TODO: DATA TYPE VALIDATION
        DataType resultDataType = DataType.getLarger(ownDataType, otherDataType);
        return new LazyTensor(this.backend, resultShape, resultDataType, () -> perform(ctx, a, b));
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient a, @NotNull IGraph.ITensorNodeWithGradient b) {
        // See: https://cs231n.github.io/optimization-2/#mat (Stanford University CS231n: Deep Learning for Computer Vision)
        // Notation: WX = D

        // W = a, X = b, D = result
        // L = loss function (or more generally, the root of the graph that we derive in respect to)
        // G = upstream gradient = dL/dD

        // .T = transpose
        // @ = matrix multiplication

        // Gradients:
        // dL/dW = G @ X.T
        // dL/dX = W.T @ G

        // TODO: OPTIMIZE

        if (a.requiresGradients()) {
            ITensor dLdW = upstreamGradient.matmul(b.getValue().transpose());
            a.accumulateGradient(dLdW);
        }

        if (b.requiresGradients()) {
            ITensor dLdX = a.getValue().transpose().matmul(upstreamGradient);
            b.accumulateGradient(dLdX);
        }
    }
}
