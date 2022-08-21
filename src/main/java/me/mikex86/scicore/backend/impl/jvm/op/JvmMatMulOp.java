package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.backend.impl.jvm.JvmTensor;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

public class JvmMatMulOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final ISciCoreBackend backend;

    public JvmMatMulOp(@NotNull ISciCoreBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull ITensor a, @NotNull ITensor b) {
        long[] shape = a.getShape();
        long[] otherShape = b.getShape();
        Validator.assertTrue(otherShape.length == 2, "Only 2D matrices are supported");
        Validator.assertTrue(shape.length == 2, "Only 2D matrices are supported");
        Validator.assertTrue(shape[1] == otherShape[0], "Shape mismatch. A.shape[1] != B.shape[0]");
        Validator.assertTrue(a.getDataType().isNumeric(), "Data type of A is not numeric");
        Validator.assertTrue(b.getDataType().isNumeric(), "Data type of B is not numeric");
        long[] resultShape = new long[]{shape[0], otherShape[1]};
        DataType ownDataType = a.getDataType();
        DataType otherDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(ownDataType, otherDataType);

        ITensor result = new JvmTensor(this.backend, resultDataType, resultShape);

        long[] index = new long[resultShape.length];
        if (resultDataType.isFloatingPoint()) {
            for (int i = 0; i < resultShape[0]; i++) {
                for (int j = 0; j < resultShape[1]; j++) {
                    double sum = 0;
                    for (int k = 0; k < shape[1]; k++) {
                        index[0] = i;
                        index[1] = k;
                        double aValue = a.getAsDouble(index);
                        index[0] = k;
                        index[1] = j;
                        double bValue = b.getAsDouble(index);
                        sum += aValue * bValue;
                    }
                    index[0] = i;
                    index[1] = j;
                    result.setByDouble(sum, index);
                }
            }
        } else {
            for (int i = 0; i < resultShape[0]; i++) {
                for (int j = 0; j < resultShape[1]; j++) {
                    long sum = 0;
                    for (int k = 0; k < shape[1]; k++) {
                        index[0] = i;
                        index[1] = k;
                        long aValue = a.getAsLong(index);
                        index[0] = k;
                        index[1] = j;
                        long bValue = b.getAsLong(index);
                        sum += aValue * bValue;
                    }
                    index[0] = i;
                    index[1] = j;
                    result.setByLong(sum, index);
                }
            }
        }
        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull ITensor a, @NotNull ITensor b) {
        long[] shape = a.getShape();
        long[] otherShape = b.getShape();
        Validator.assertTrue(otherShape.length == 2, "Only 2D matrices are supported");
        Validator.assertTrue(shape.length == 2, "Only 2D matrices are supported");
        Validator.assertTrue(shape[1] == otherShape[0], "Shape mismatch. A.shape[1] != B.shape[0]");
        Validator.assertTrue(a.getDataType().isNumeric(), "Data type of A is not numeric");
        Validator.assertTrue(b.getDataType().isNumeric(), "Data type of B is not numeric");
        long[] resultShape = new long[]{shape[0], otherShape[1]};
        DataType ownDataType = a.getDataType();
        DataType otherDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(ownDataType, otherDataType);
        return new LazyTensor(this.backend, resultShape, resultDataType, () -> perform(a, b));
    }

    @Override
    public void computeGradients(@NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient a, @NotNull IGraph.ITensorNodeWithGradient b) {
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
