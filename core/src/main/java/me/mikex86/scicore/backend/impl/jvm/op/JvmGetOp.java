package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.op.IDifferentiableOperation;
import me.mikex86.scicore.graph.op.IOperation;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.tensor.View;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.util.*;

public class JvmGetOp implements IOperation, IDifferentiableOperation {

    @NotNull
    private final JvmBackend backend;

    public JvmGetOp(@NotNull JvmBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull List<ITensor> inputs) {
        ITensor input = inputs.get(0);
        if (inputs.size() < 2) {
            throw new IllegalArgumentException("Get operation must be supplied at least one index!");
        }
        List<ITensor> indicesList = inputs.subList(1, inputs.size());
        long[] indexShape = indicesList.get(0).getShape();
        for (ITensor index : indicesList) {
            if (!index.getDataType().isInteger()) {
                throw new IllegalArgumentException("Index tensor must be an integer type");
            }
            indexShape = ShapeUtils.broadcastShapes(indexShape, index.getShape());
        }
        long[] inputShape = input.getShape();
        int nIndices = indicesList.size();
        if (nIndices > inputShape.length) {
            throw new IllegalArgumentException("Too many indices for get operation");
        }
        long[] resultShape = new long[indexShape.length + inputShape.length - nIndices];
        System.arraycopy(indexShape, 0, resultShape, 0, indexShape.length);
        System.arraycopy(inputShape, nIndices, resultShape, indexShape.length, inputShape.length - nIndices);
        ITensor result = backend.createTensor(input.getDataType(), resultShape);

        long[] indexIntoIndexTensor = new long[indexShape.length];
        long[] indexIntoIndexTensorConstrained = new long[indexShape.length];
        long flatIndex = 0;
        do {
            ITensor currentView = input;
            for (ITensor index : indicesList) {
                System.arraycopy(indexIntoIndexTensor, 0, indexIntoIndexTensorConstrained, 0, indexIntoIndexTensorConstrained.length);
                ShapeUtils.constrainIndex(indexIntoIndexTensorConstrained, index.getShape());
                long indexValue = index.getAsLong(indexIntoIndexTensorConstrained);
                if (indexValue < 0 || indexValue >= currentView.getShape()[0]) {
                    throw new IllegalArgumentException("Index out of bounds");
                }
                currentView = currentView.getView(indexValue);
            }
            result.setContentsWithOffset(flatIndex, currentView);
            flatIndex += currentView.getNumberOfElements();
        } while (ShapeUtils.incrementIndex(indexIntoIndexTensor, indexShape));
        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull List<ITensor> inputs) {
        ITensor input = inputs.get(0);
        long[] inputShape = input.getShape();
        if (inputs.size() < 2) {
            throw new IllegalArgumentException("Get operation must be supplied at least one index!");
        }
        List<ITensor> indicesList = inputs.subList(1, inputs.size());
        long[] indexShape = indicesList.get(0).getShape();
        for (ITensor index : indicesList) {
            if (!index.getDataType().isInteger()) {
                throw new IllegalArgumentException("Index tensor must be an integer type");
            }
            indexShape = ShapeUtils.broadcastShapes(indexShape, index.getShape());
        }
        int nIndices = indicesList.size();
        if (nIndices > inputShape.length) {
            throw new IllegalArgumentException("Too many indices for get operation");
        }
        long[] resultShape = new long[indexShape.length + inputShape.length - nIndices];
        System.arraycopy(indexShape, 0, resultShape, 0, indexShape.length);
        System.arraycopy(inputShape, nIndices, resultShape, indexShape.length, inputShape.length - nIndices);
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

            long[] indexShape = indicesList.get(0).getShape();
            for (ITensor index : indicesList) {
                if (!index.getDataType().isInteger()) {
                    throw new IllegalArgumentException("Index tensor must be an integer type");
                }
                indexShape = ShapeUtils.broadcastShapes(indexShape, index.getShape());
            }

            ITensor gradient = backend.createTensor(input.getDataType(), input.getShape());

            long[] indexTensorIndex = new long[indexShape.length];
            long[] indexTensorConstrained = new long[indexShape.length];
            long[] indexIntoGradient = null;

            // used to check for duplicate indices
            // When an index is duplicated, multiple gradient contributions need to be accumulated,
            // which means we cannot use fast .setContents() but slower .add()
            Set<Long> visitedOffsets = new HashSet<>();
            do {
                long offset = 0;
                long[] index = new long[indicesList.size()];
                for (int dim = 0; dim < indicesList.size(); dim++) {
                    ITensor indexDimTensor = indicesList.get(dim);
                    System.arraycopy(indexTensorIndex, 0, indexTensorConstrained, 0, indexTensorConstrained.length);
                    ShapeUtils.constrainIndex(indexTensorConstrained, indexDimTensor.getShape());
                    long indexValue = indexDimTensor.getAsLong(indexTensorConstrained);
                    if (indexValue < 0 || indexValue >= gradient.getShape()[dim]) {
                        throw new IllegalArgumentException("Index out of bounds");
                    }
                    index[dim] = indexValue;
                    offset += indexValue * gradient.getStrides()[dim];
                }
                ITensor gradientView = gradient.getView(index);
                if (indexIntoGradient == null) {
                    indexIntoGradient = new long[upstreamGradient.getShape().length - gradientView.getShape().length];
                }
                if (visitedOffsets.contains(offset)) {
                    ITensor src = upstreamGradient.getView(indexIntoGradient);
                    gradientView.add(src);
                } else {
                    gradientView.setContents(upstreamGradient.getView(indexIntoGradient));
                    visitedOffsets.add(offset);
                }
                ShapeUtils.incrementIndex(indexIntoGradient, upstreamGradient.getShape());
            } while (ShapeUtils.incrementIndex(indexTensorIndex, indexShape));

            inputNode.accumulateGradient(gradient);
        }
    }
}
