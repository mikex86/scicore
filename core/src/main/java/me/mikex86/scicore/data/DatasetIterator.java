package me.mikex86.scicore.data;

import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.utils.Pair;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.util.*;
import java.util.function.Supplier;

public class DatasetIterator {

    private final int batchSize;

    /**
     * Returns a random sample (X, Y) of the dataset.
     */
    @NotNull
    private final Supplier<@NotNull Pair<@NotNull ITensor, @NotNull ITensor>> dataProvider;

    public DatasetIterator(int batchSize, @NotNull Supplier<@NotNull Pair<@NotNull ITensor, @NotNull ITensor>> dataProvider) {
        this.batchSize = batchSize;
        this.dataProvider = dataProvider;
    }

    public int getBatchSize() {
        return batchSize;
    }

    @NotNull
    public Pair<ITensor, ITensor> next() {
        Pair<ITensor, ITensor> firstEntry = dataProvider.get();

        ITensor firstTensor = firstEntry.getFirst();
        ITensor firstLabel = firstEntry.getSecond();

        ISciCoreBackend backend = firstTensor.getSciCoreBackend();
        long[] dataShape = firstTensor.getShape();
        long[] labelShape = firstLabel.getShape();

        long[] batchedDataShape = new long[dataShape.length + 1];
        System.arraycopy(dataShape, 0, batchedDataShape, 1, dataShape.length);
        batchedDataShape[0] = batchSize;

        long[] batchedLabelShape = new long[labelShape.length + 1];
        System.arraycopy(labelShape, 0, batchedLabelShape, 1, labelShape.length);
        batchedLabelShape[0] = batchSize;

        ITensor batchedX = backend.createTensor(firstTensor.getDataType(), batchedDataShape);
        ITensor batchedY = backend.createTensor(firstLabel.getDataType(), batchedLabelShape);

        // copy data from batch to final tensor
        for (int i = 0; i < batchSize; i++) {
            Pair<ITensor, ITensor> entry;
            {
                if (firstEntry != null) {
                    entry = firstEntry;
                    firstEntry = null;
                } else {
                    entry = dataProvider.get();
                }
            }
            ITensor X = entry.getFirst();
            ITensor Y = entry.getSecond();
            long[] xShape = X.getShape();
            if (!ShapeUtils.equals(xShape, dataShape)) {
                throw new IllegalArgumentException("Tensor shape mismatch");
            }
            long[] yShape = Y.getShape();
            if (!ShapeUtils.equals(yShape, labelShape)) {
                throw new IllegalArgumentException("Tensor shape mismatch");
            }
            batchedX.setContents(new long[]{i}, X, true);
            batchedY.setContents(new long[]{i}, Y, true);
        }
        return Pair.of(batchedX, batchedY);
    }
}
