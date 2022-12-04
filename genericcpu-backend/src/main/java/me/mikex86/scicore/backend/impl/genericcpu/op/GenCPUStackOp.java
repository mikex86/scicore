package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.OptionBundle;
import me.mikex86.scicore.graph.op.IDifferentiableOperation;
import me.mikex86.scicore.graph.op.IOperation;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class GenCPUStackOp implements IOperation, IDifferentiableOperation {

    private final GenCPUBackend backend;

    public GenCPUStackOp(GenCPUBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(Graph.@NotNull IOperationContext ctx, @NotNull List<@NotNull ITensor> inputs) {
        OptionBundle optionBundle = ctx.getOptionBundle();
        int dimension = optionBundle.getInt("dimension").orElseThrow();

        ITensor first = inputs.get(0);
        long[] stackedShape = first.getShape();

        for (int i = 1; i < inputs.size(); i++) {
            ITensor tensor = inputs.get(i);
            long[] shape = tensor.getShape();
            if (!ShapeUtils.equals(shape, stackedShape)) {
                throw new IllegalArgumentException("All tensors must have the same shape");
            }
            if (tensor.getDataType() != first.getDataType()) {
                throw new IllegalArgumentException("All tensors must have the same data type");
            }
        }

        long[] resultShape = new long[stackedShape.length + 1];
        for (int i = 0; i < stackedShape.length; i++) {
            resultShape[i < dimension ? i : i + 1] = stackedShape[i];
        }
        resultShape[dimension] = inputs.size();

        ITensor result = backend.createTensor(inputs.get(0).getDataType(), resultShape);

        long[] tensorIndex = new long[stackedShape.length];
        long[] resultIndex = new long[resultShape.length];

        // TODO: MOVE THIS TO JNI
        for (int i = 0; i < inputs.size(); i++) {
            ITensor tensor = inputs.get(i);
            Arrays.fill(tensorIndex, 0);
            resultIndex[dimension] = i;
            do {
                for (int j = 0; j < tensorIndex.length; j++) {
                    resultIndex[j < dimension ? j : j + 1] = tensorIndex[j];
                }
                if (result.getDataType().isFloatingPoint()) {
                    result.setByDouble(tensor.getAsDouble(tensorIndex), resultIndex);
                } else {
                    result.setByLong(tensor.getAsLong(tensorIndex), resultIndex);
                }
            } while (ShapeUtils.incrementIndex(tensorIndex, stackedShape));
        }
        return result;
    }

    @Override
    public @NotNull ITensor performLazily(Graph.@NotNull IOperationContext ctx, @NotNull List<@NotNull ITensor> inputs) {
        OptionBundle optionBundle = ctx.getOptionBundle();
        int dimension = optionBundle.getInt("dimension").orElseThrow();

        ITensor first = inputs.get(0);
        long[] stackedShape = first.getShape();

        for (int i = 1; i < inputs.size(); i++) {
            ITensor tensor = inputs.get(i);
            long[] shape = tensor.getShape();
            if (!ShapeUtils.equals(shape, stackedShape)) {
                throw new IllegalArgumentException("All tensors must have the same shape");
            }
            if (tensor.getDataType() != first.getDataType()) {
                throw new IllegalArgumentException("All tensors must have the same data type");
            }
        }

        long[] resultShape = new long[stackedShape.length + 1];
        for (int i = 0; i < stackedShape.length; i++) {
            resultShape[i < dimension ? i : i + 1] = stackedShape[i];
        }
        resultShape[dimension] = inputs.size();
        return new LazyTensor(backend, resultShape, first.getDataType());
    }

    @Override
    public void computeGradients(Graph.@NotNull OperationGraphNode operationNode) {
        OptionBundle optionBundle = operationNode.getOperationContext().getOptionBundle();
        int dimension = optionBundle.getInt("dimension").orElseThrow();

        List<IGraph.IGraphNode> inputs = operationNode.getInputs();

        // TODO: MOVE THIS TO JNI
        long[] resultShape = operationNode.getValue().getShape();
        try (ITensor resultGradient = Objects.requireNonNull(operationNode.getUpstreamGradient()).broadcast(resultShape)) {
            long[] stackedShape = ((IGraph.ITensorNode) inputs.get(0)).getValue().getShape();

            long[] tensorIndex = new long[stackedShape.length];
            long[] resultIndex = new long[resultShape.length];

            for (int i = 0; i < inputs.size(); i++) {
                IGraph.ITensorNodeWithGradient input = (IGraph.ITensorNodeWithGradient) inputs.get(i);
                if (!input.requiresGradients()) {
                    continue;
                }
                resultIndex[dimension] = i;

                ITensor inputGradient = backend.createTensor(resultGradient.getDataType(), stackedShape);
                Arrays.fill(tensorIndex, 0);
                do {
                    for (int j = 0; j < tensorIndex.length; j++) {
                        resultIndex[j < dimension ? j : j + 1] = tensorIndex[j];
                    }
                    if (resultGradient.getDataType().isFloatingPoint()) {
                        inputGradient.setByDouble(resultGradient.getAsDouble(resultIndex), tensorIndex);
                    } else {
                        inputGradient.setByLong(resultGradient.getAsLong(resultIndex), tensorIndex);
                    }
                } while (ShapeUtils.incrementIndex(tensorIndex, stackedShape));

                input.accumulateGradient(inputGradient);
            }
        }
    }
}
