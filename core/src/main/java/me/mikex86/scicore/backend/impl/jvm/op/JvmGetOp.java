package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.op.IDifferentiableOperation;
import me.mikex86.scicore.graph.op.IOperation;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.List;

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
        for (ITensor index : indicesList) {
            if (!index.getDataType().isInteger()) {
                throw new IllegalArgumentException("Index tensor must be an integer type");
            }
        }
        ITensor firstIndex = indicesList.get(0);
        long[] inputShape = input.getShape();
        long[] indicesShape = firstIndex.getShape();
        int nIndices = indicesList.size();
        long[] resultShape = new long[indicesShape.length + inputShape.length - nIndices];
        System.arraycopy(indicesShape, 0, resultShape, 0, indicesShape.length);
        System.arraycopy(inputShape, nIndices, resultShape, indicesShape.length, inputShape.length - nIndices);
        ITensor result = backend.createTensor(input.getDataType(), resultShape);

        // for every dimension dim in indicesList.get(dim), we store the index into that index tensor that we currently are reading
        List<long[]> indicesIntoIndices = new ArrayList<>(indicesList.size());
        for (int i = 0; i < indicesList.size(); i++) {
            indicesIntoIndices.add(new long[indicesShape.length]);
        }
        loop:
        {
            while (true) {
                ITensor inputView = input;
                for (int dim = 0; dim < indicesList.size(); dim++) {
                    ITensor indexTensor = indicesList.get(dim);
                    long[] indexIntoIndex = indicesIntoIndices.get(dim);
                    long indexIntoInput = indexTensor.getAsLong(indexIntoIndex);
                    if (indexIntoInput < 0 || indexIntoInput >= inputShape[dim]) {
                        throw new IllegalArgumentException("Index " + indexIntoInput + " is out of bounds for shape " + ShapeUtils.toString(inputShape));
                    }
                    inputView = inputView.getView(indexIntoInput); // cheap
                    if (dim == indicesList.size() - 1) {
                        result.setContents(indexIntoIndex, inputView); // mem-copy
                    }
                    boolean incrementIndex = ShapeUtils.incrementIndex(indexIntoIndex, indicesShape);
                    if (!incrementIndex && dim == 0) {
                        break loop;
                    }
                }
            }
        }
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
        long[] resultShape = new long[indicesShape.length + inputShape.length - nIndices];
        System.arraycopy(indicesShape, 0, resultShape, 0, indicesShape.length);
        System.arraycopy(inputShape, nIndices, resultShape, indicesShape.length, inputShape.length - nIndices);
        return new LazyTensor(backend, resultShape, input.getDataType());
    }

    @Override
    public void computeGradients(@NotNull Graph.OperationGraphNode operationNode) {
        throw new UnsupportedOperationException("Not implemented yet");
    }
}
