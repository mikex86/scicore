package me.mikex86.scicore.backend.impl.jvm.op;

import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.op.IOperation;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public class JvmGetOp implements IOperation {

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
        long[] indexIntoIndex = new long[indicesShape.length];
        int nIndices = indicesList.size();
        long[] resultShape = new long[indicesShape.length + inputShape.length - nIndices];
        System.arraycopy(indicesShape, 0, resultShape, 0, indicesShape.length);
        System.arraycopy(inputShape, nIndices, resultShape, indicesShape.length, inputShape.length - nIndices);
        ITensor result = backend.createTensor(input.getDataType(), resultShape);
        do {
            ITensor inputView = input;
            for (int dim = 0; dim < indicesList.size(); dim++) {
                ITensor index = indicesList.get(dim);
                long dimIndex = index.getAsLong(indexIntoIndex);
                if (dimIndex < 0 || dimIndex >= inputShape[dim]) {
                    throw new IllegalArgumentException("Index " + index + " is out of bounds for shape " + ShapeUtils.toString(inputShape));
                }
                inputView = input.getView(dimIndex); // cheap
            }
            result.setContents(indexIntoIndex, inputView); // mem-copy
        } while (ShapeUtils.incrementIndex(indexIntoIndex, indicesShape));
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
}
