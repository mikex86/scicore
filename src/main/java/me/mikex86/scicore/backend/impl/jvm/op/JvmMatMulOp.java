package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.Tensor;
import me.mikex86.scicore.backend.ITensorImpl;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.backend.impl.jvm.JvmDataTensorImpl;
import me.mikex86.scicore.backend.impl.jvm.JvmDerivedTensor;
import me.mikex86.scicore.op.IBinaryOperation;
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
        long[] otherShape = b.getShape();
        if (otherShape.length != 2) {
            throw new IllegalArgumentException("matmul only supports 2D matrices");
        }
        long[] shape = a.getShape();
        if (shape.length != 2) {
            throw new IllegalArgumentException("matmul only supports 2D matrices");
        }
        if (shape[1] != otherShape[0]) {
            throw new IllegalArgumentException("matmul: shape mismatch. A.shape[1] != B.shape[0]");
        }
        long[] resultShape = new long[]{shape[0], otherShape[1]};
        DataType ownDataType = a.getDataType();
        DataType otherDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(ownDataType, otherDataType);

        ITensorImpl result = new JvmDataTensorImpl(this.backend, resultDataType, resultShape);
        ITensor resultTensor = new Tensor(this.backend, result);

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
                    resultTensor.setByDouble(sum, index);
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
                    resultTensor.setByLong(sum, index);
                }
            }
        }
        return resultTensor;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull ITensor a, @NotNull ITensor b) {
        long[] otherShape = b.getShape();
        Validator.assertTrue(otherShape.length == 2, "matmul only supports 2D matrices");
        long[] shape = a.getShape();
        if (shape.length != 2) {
            throw new IllegalArgumentException("matmul only supports 2D matrices");
        }
        if (shape[1] != otherShape[0]) {
            throw new IllegalArgumentException("matmul: shape mismatch. A.shape[1] != B.shape[0]");
        }
        long[] resultShape = new long[]{shape[0], otherShape[1]};
        DataType ownDataType = a.getDataType();
        DataType otherDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(ownDataType, otherDataType);
        return new JvmDerivedTensor(this.backend, resultShape, resultDataType, () -> perform(a, b));
    }

    @Override
    public void computeGradients(@NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient a, @NotNull IGraph.ITensorNodeWithGradient b) {
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
