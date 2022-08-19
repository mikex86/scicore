package me.mikex86.scicore.backend.impl.jvm;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.backend.ITensorImpl;
import me.mikex86.scicore.backend.impl.jvm.op.*;
import me.mikex86.scicore.op.*;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

import java.util.List;
import java.util.Map;

public class JvmBackend implements ISciCoreBackend {

    @NotNull
    private final IGraphRecorder operationRecorder = new GraphRecorder(this);

    @NotNull
    private final Map<OperationType, IOperation> operationTable = Map.of(
            OperationType.MATMUL, new JvmMatMulOp(this),
            OperationType.DIVIDED, new JvmDividedOp(this),
            OperationType.PLUS, new JvmPlusOp(this),
            OperationType.REDUCE_SUM, new JvmReduceSumOp(this),
            OperationType.EXP, new JvmExpOp(this),
            OperationType.TRANSPOSE, new JvmTransposeOp(this)
    );

    @Override
    public @NotNull ITensorImpl createTensor(@NotNull DataType dataType, long @NotNull [] shape) {
        return new JvmDataTensorImpl(this, dataType, shape);
    }

    @Override
    public @NotNull IGraphRecorder getOperationRecorder() {
        return this.operationRecorder;
    }

    public @NotNull IOperation getOperation(@NotNull OperationType operationType) {
        IOperation operation = operationTable.get(operationType);
        if (operation == null) {
            throw new IllegalArgumentException("Operation not implemented for JVMBackend: " + operationType);
        }
        return operation;
    }

    @Override
    @SuppressWarnings({"rawtypes", "unchecked"})
    public @NotNull ITensor lazyOpTensor(@NotNull IOperation operation, @NotNull List<@NotNull Object> inputs) {
        if (operation instanceof IUnaryOperation unaryOperation) {
            Validator.assertTrue(inputs.size() == 1, "Unary operation expects exactly one input.");
            Validator.assertTrue(inputs.get(0) instanceof ITensor, "inputs[0] must be a tensor for unary operation.");
            return unaryOperation.performLazily((ITensor) inputs.get(0));
        } else if (operation instanceof IBinaryOperation binaryOperation) {
            Validator.assertTrue(inputs.size() == 2, "Binary operation expects exactly two inputs");
            Validator.assertTrue(inputs.get(0) instanceof ITensor, "inputs[0] must be a tensor for binary operation.");
            Validator.assertTrue(inputs.get(1) instanceof ITensor, "inputs[1] must be a tensor for binary operation.");
            return binaryOperation.performLazily((ITensor) inputs.get(0), (ITensor) inputs.get(1));
        } else if (operation instanceof IParametricOperation) {
            if (operation instanceof IBiParametricOperation biParametricOperation) {
                Validator.assertTrue(inputs.size() == 3, "Bi-parametric operation expects one tensor and two arguments (3 in total).");
                Validator.assertTrue(inputs.get(0) instanceof ITensor, "inputs[0] must be a tensor for binary operation.");
                //noinspection unchecked
                return biParametricOperation.performLazily((ITensor) inputs.get(0), inputs.get(1), inputs.get(2));
            } else {
                throw new IllegalStateException("Unknown parametric operation type: " + operation.getClass().getSimpleName());
            }
        } else {
            throw new IllegalStateException("Unknown operation type: " + operation.getClass().getSimpleName());
        }
    }

    @Override
    public void computeGradients(@NotNull Graph.OperationGraphNode operationNode) {
        OperationType operationType = operationNode.getOperationType();
        IOperation operation = getOperation(operationType);
        if (!(operation instanceof IDifferentiableOperation differentiableOperation)) {
            throw new IllegalStateException("Operation is not differentiable: " + operationType);
        }
        differentiableOperation.computeGradients(operationNode);
    }
}
