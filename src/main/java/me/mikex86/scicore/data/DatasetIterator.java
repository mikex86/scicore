package me.mikex86.scicore.data;

import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.Tensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.backend.ITensorImpl;
import org.jetbrains.annotations.NotNull;

import java.util.*;

public class DatasetIterator implements Iterator<ITensor> {

    private final int batchSize;

    @NotNull
    private final Iterator<@NotNull ITensor> dataProvider;

    @NotNull
    private final Queue<@NotNull ITensor> batch;

    public DatasetIterator(int batchSize, @NotNull Iterator<ITensor> dataProvider) {
        this.batchSize = batchSize;
        this.dataProvider = dataProvider;
        this.batch = new LinkedList<>();
    }

    public int getBatchSize() {
        return batchSize;
    }

    public boolean hasNext() {
        // fill batch and determine if there is a next batch
        for (int i = 0; i < batchSize; i++) {
            if (!dataProvider.hasNext()) {
                return false;
            }
            batch.add(dataProvider.next());
        }
        return true;
    }

    @NotNull
    public ITensor next() {
        if (!hasNext()) {
            throw new IllegalStateException("No next batch");
        }
        ITensor firstTensor = Objects.requireNonNull(batch.peek());
        ISciCoreBackend backend = firstTensor.getSciCoreBackend();
        long[] dataShape = firstTensor.getShape();
        long[] finalShape = new long[dataShape.length + 1];

        ITensorImpl finalTensor = backend.createTensor(firstTensor.getDataType(), finalShape);

        // copy data from batch to final tensor
        for (int i = 0; i < batchSize; i++) {
            ITensor tensor = Objects.requireNonNull(batch.poll(), "Could not poll tensor from fetched batch in DatasetIterator");
            long[] tensorShape = tensor.getShape();
            if (!Arrays.equals(tensorShape, dataShape)) {
                throw new IllegalArgumentException("Tensor shape mismatch");
            }
            finalTensor.setContents(new long[]{i}, tensor, true);
        }

        return new Tensor(backend, finalTensor);
    }
}
