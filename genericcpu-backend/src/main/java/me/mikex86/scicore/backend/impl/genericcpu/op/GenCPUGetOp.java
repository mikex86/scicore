package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.op.IDifferentiableOperation;
import me.mikex86.scicore.graph.op.IOperation;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.util.List;
import java.util.Objects;

public class GenCPUGetOp implements IOperation, IDifferentiableOperation {

    @NotNull
    private final GenCPUBackend backend;

    public GenCPUGetOp(@NotNull GenCPUBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull List<ITensor> inputs) {
        ITensor input = inputs.get(0);
        if (inputs.size() < 2) {
            throw new IllegalArgumentException("Get operation must be supplied at least one index!");
        }
        List<ITensor> indicesList = inputs.subList(1, inputs.size());
        for (ITensor index : indicesList) {
            if (!index.getDataType().isInteger()) {
                throw new IllegalArgumentException("Index tensor must be an integer type");
            }
            if (!ShapeUtils.equals(index.getShape(), indicesList.get(0).getShape())) {
                throw new IllegalArgumentException("Index tensors for get operation must have the same shape");
            }
        }
        ITensor firstIndex = indicesList.get(0);
        long[] inputShape = input.getShape();
        long[] firstIndexShape = firstIndex.getShape();
        int nIndices = indicesList.size();
        if (nIndices > inputShape.length) {
            throw new IllegalArgumentException("Too many indices for get operation");
        }
        long[] resultShape = new long[firstIndexShape.length + inputShape.length - nIndices];
        System.arraycopy(firstIndexShape, 0, resultShape, 0, firstIndexShape.length);
        System.arraycopy(inputShape, nIndices, resultShape, firstIndexShape.length, inputShape.length - nIndices);
        ITensor result = backend.createTensor(input.getDataType(), resultShape);

        // TODO: MOVE THIS TO JNI
        long[] indexIntoFirstIndex = new long[firstIndexShape.length];
        long flatIndex = 0;
        do {
            ITensor currentView = input;
            for (ITensor index : indicesList) {
                long indexValue = index.getAsLong(indexIntoFirstIndex);
                if (indexValue < 0 || indexValue >= currentView.getShape()[0]) {
                    throw new IllegalArgumentException("Index out of bounds");
                }
                currentView = currentView.getView(indexValue);
            }
            result.setContentsWithOffset(flatIndex, currentView);
            flatIndex += currentView.getNumberOfElements();
        } while (ShapeUtils.incrementIndex(indexIntoFirstIndex, firstIndexShape));
        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull List<ITensor> inputs) {
        ITensor input = inputs.get(0);
        ITensor indices = inputs.get(1);
        long[] inputShape = input.getShape();
        long[] indicesShape = indices.getShape();
        if (inputs.size() < 2) {
            throw new IllegalArgumentException("Get operation must be supplied at least one index!");
        }
        List<ITensor> indicesList = inputs.subList(1, inputs.size());
        for (ITensor index : indicesList) {
            if (!index.getDataType().isInteger()) {
                throw new IllegalArgumentException("Index tensor must be an integer type");
            }
        }
        int nIndices = indicesList.size();
        if (nIndices > inputShape.length) {
            throw new IllegalArgumentException("Too many indices for get operation");
        }
        long[] resultShape = new long[indicesShape.length + inputShape.length - nIndices];
        System.arraycopy(indicesShape, 0, resultShape, 0, indicesShape.length);
        System.arraycopy(inputShape, nIndices, resultShape, indicesShape.length, inputShape.length - nIndices);
        return new LazyTensor(backend, resultShape, input.getDataType());
    }

    @Override
    public void computeGradients(@NotNull Graph.OperationGraphNode operationNode) {
        List<IGraph.IGraphNode> inputs = operationNode.getInputs();
        IGraph.ITensorNodeWithGradient inputNode = (IGraph.ITensorNodeWithGradient) inputs.get(0);
        if (inputNode.requiresGradients()) {
            ITensor upstreamGradient = operationNode.getUpstreamGradient();
            Objects.requireNonNull(upstreamGradient, "Upstream gradient must not be null");
            List<ITensor> indicesList = inputs.subList(1, inputs.size()).stream()
                    .map(node -> ((IGraph.ITensorNodeWithGradient) node).getValue())
                    .toList();
            ITensor input = inputNode.getValue();

            ITensor firstIndex = indicesList.get(0);
            long[] firstIndexShape = firstIndex.getShape();

            ITensor gradient = backend.createTensor(input.getDataType(), input.getShape());

            // TODO: MOVE THIS TO JNI
            long[] indexIntoFirstIndex = new long[firstIndexShape.length];
            long[] indexIntoGradient = null;
            do {
                ITensor currentView = gradient;
                for (ITensor index : indicesList) {
                    long indexValue = index.getAsLong(indexIntoFirstIndex);
                    if (indexValue < 0 || indexValue >= currentView.getShape()[0]) {
                        throw new IllegalArgumentException("Index out of bounds");
                    }
                    currentView = currentView.getView(indexValue);
                }
                if (indexIntoGradient == null) {
                    indexIntoGradient = new long[upstreamGradient.getShape().length - currentView.getShape().length];
                }
                currentView.add(upstreamGradient.getView(indexIntoGradient));
                ShapeUtils.incrementIndex(indexIntoGradient, upstreamGradient.getShape());
            } while (ShapeUtils.incrementIndex(indexIntoFirstIndex, firstIndexShape));

            inputNode.accumulateGradient(gradient);
        }
    }
}
