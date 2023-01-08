package me.mikex86.scicore.backend.impl.genericcpu.op;

import me.mikex86.scicore.backend.impl.genericcpu.GenCPUBackend;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graph.OptionBundle;
import me.mikex86.scicore.graph.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import me.mikex86.scicore.tensor.View;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;


public class GenCPUConcatOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final GenCPUBackend backend;

    public GenCPUConcatOp(@NotNull GenCPUBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        OptionBundle optionBundle = ctx.getOptionBundle();
        int dimension = optionBundle.getInt("dimension").orElseThrow();
        DataType aDataType = a.getDataType();
        DataType bDataType = b.getDataType();
        if (!aDataType.equals(bDataType)) {
            throw new IllegalArgumentException("Concatenation of tensors with different data types is not supported");
        }
        long[] aShape = a.getShape();
        long[] bShape = b.getShape();
        if (aShape.length != bShape.length) {
            throw new IllegalArgumentException("Tensors must have the same number of dimensions");
        }
        for (int i = 0; i < aShape.length; i++) {
            if (i != dimension && aShape[i] != bShape[i]) {
                throw new IllegalArgumentException("Tensors must have the same shape except for the dimension to concatenate");
            }
        }
        long[] resultShape = new long[aShape.length];
        System.arraycopy(aShape, 0, resultShape, 0, aShape.length);
        resultShape[dimension] += bShape[dimension];
        return new LazyTensor(backend, resultShape, aDataType);
    }

    @Override
    public @NotNull ITensor perform(Graph.@NotNull IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        OptionBundle optionBundle = ctx.getOptionBundle();

        int dimension = optionBundle.getInt("dimension").orElseThrow();

        DataType aDataType = a.getDataType();
        DataType bDataType = b.getDataType();

        if (!aDataType.equals(bDataType)) {
            throw new IllegalArgumentException("Concatenation of tensors with different data types is not supported");
        }

        long[] aShape = a.getShape();
        long[] bShape = b.getShape();

        if (aShape.length != bShape.length) {
            throw new IllegalArgumentException("Tensors must have the same number of dimensions");
        }

        for (int i = 0; i < aShape.length; i++) {
            if (i != dimension && aShape[i] != bShape[i]) {
                throw new IllegalArgumentException("Tensors must have the same shape except for the dimension to concatenate");
            }
        }

        long[] aStrides = a.getStrides();
        long[] bStrides = b.getStrides();

        long[] resultShape = new long[aShape.length];
        System.arraycopy(aShape, 0, resultShape, 0, aShape.length);
        resultShape[dimension] += bShape[dimension];

        ITensor result = backend.createTensor(aDataType, resultShape);

        long[] resultStrides = result.getStrides();

        if (dimension == 0) {
            result.setContentsWithOffset(0, a);
            long numElementsA = a.getNumberOfElements();
            result.setContentsWithOffset(numElementsA, b);
        } else {
            int superDimension = dimension - 1;
            long nElementsUptoSuperDim = ShapeUtils.getNumElementsOfMostSignificantDims(aShape, superDimension);

            long numElementsInConcatDimensionInA = ShapeUtils.getNumElementsOfLeastSignificantDims(aShape, aShape.length - dimension);
            long numElementsInConcatDimensionInB = ShapeUtils.getNumElementsOfLeastSignificantDims(bShape, bShape.length - dimension);
            long numElementsInConcatDimensionInResult = ShapeUtils.getNumElementsOfLeastSignificantDims(resultShape, resultShape.length - dimension);
            assert numElementsInConcatDimensionInResult == numElementsInConcatDimensionInA + numElementsInConcatDimensionInB;

            long superDimStrideInResult = resultStrides[superDimension];
            for (int i = 0; i < nElementsUptoSuperDim; i++) {
                long offsetInA = i * aStrides[superDimension];
                long offsetInB = i * bStrides[superDimension];
                long offsetInResult = i * superDimStrideInResult;

                View aView = new View(a.getSciCoreBackend(), a.getDataContainer(), new long[]{numElementsInConcatDimensionInA}, offsetInA, Arrays.copyOfRange(aStrides, superDimension + 1, aStrides.length));
                View bView = new View(b.getSciCoreBackend(), b.getDataContainer(), new long[]{numElementsInConcatDimensionInB}, offsetInB, Arrays.copyOfRange(bStrides, superDimension + 1, bStrides.length));
                result.setContentsWithOffset(offsetInResult, aView);
                result.setContentsWithOffset(offsetInResult + numElementsInConcatDimensionInA, bView);
            }
        }

        return result;
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient a, @NotNull IGraph.ITensorNodeWithGradient b) {
        OptionBundle optionBundle = ctx.getOptionBundle();
        int dimension = optionBundle.getInt("dimension").orElseThrow();
        if (dimension == 0) {
            ITensor aValue = a.getValue();
            if (a.requiresGradients()) {
                View upstreamGradientView = new View(upstreamGradient.getSciCoreBackend(), upstreamGradient.getDataContainer(), aValue.getShape(), 0, ShapeUtils.makeStrides(aValue.getShape()));
                a.accumulateGradient(upstreamGradientView);
            }
            if (b.requiresGradients()) {
                ITensor bValue = b.getValue();
                View upstreamGradientView = new View(upstreamGradient.getSciCoreBackend(), upstreamGradient.getDataContainer(), bValue.getShape(), aValue.getNumberOfElements(), ShapeUtils.makeStrides(bValue.getShape()));
                b.accumulateGradient(upstreamGradientView);
            }
        } else {
            if (a.requiresGradients() || b.requiresGradients()) {
                long[] upstreamGradientShape = upstreamGradient.getShape();
                long[] upstreamGradientStrides = upstreamGradient.getStrides();

                ITensor aValue = a.getValue();
                ITensor bValue = b.getValue();

                long[] aShape = aValue.getShape();
                long[] bShape = bValue.getShape();

                long[] aStrides = aValue.getStrides();
                long[] bStrides = bValue.getStrides();

                ITensor aGradient = null;
                if (a.requiresGradients()) {
                    aGradient = backend.createTensor(aValue.getDataType(), aShape);
                }
                ITensor bGradient = null;
                if (b.requiresGradients()) {
                    bGradient = backend.createTensor(bValue.getDataType(), bShape);
                }

                int superDimension = dimension - 1;
                long nElementsUptoSuperDim = ShapeUtils.getNumElementsOfMostSignificantDims(aShape, superDimension);

                long numElementsInConcatDimensionInA = ShapeUtils.getNumElementsOfLeastSignificantDims(aShape, aShape.length - dimension);
                long numElementsInConcatDimensionInB = ShapeUtils.getNumElementsOfLeastSignificantDims(bShape, bShape.length - dimension);
                long numElementsInConcatDimensionInUpstreamGradient = ShapeUtils.getNumElementsOfLeastSignificantDims(upstreamGradientShape, upstreamGradientShape.length - dimension);

                assert numElementsInConcatDimensionInA + numElementsInConcatDimensionInB == numElementsInConcatDimensionInUpstreamGradient;

                long superDimStrideInUpstreamGradient = upstreamGradientShape[superDimension];
                for (int i = 0; i < nElementsUptoSuperDim; i++) {
                    long offsetInA = i * aStrides[superDimension];
                    long offsetInB = i * bStrides[superDimension];
                    long offsetInUpstreamGradient = i * superDimStrideInUpstreamGradient;

                    if (aGradient != null) {
                        View aGradientView = new View(aGradient.getSciCoreBackend(), aGradient.getDataContainer(), new long[]{numElementsInConcatDimensionInA}, offsetInA, Arrays.copyOfRange(aStrides, superDimension + 1, aStrides.length));
                        View upstreamGradientViewForA = new View(
                                upstreamGradient.getSciCoreBackend(), upstreamGradient.getDataContainer(), new long[]{numElementsInConcatDimensionInA},
                                offsetInUpstreamGradient,
                                Arrays.copyOfRange(upstreamGradientStrides, superDimension + 1, upstreamGradientStrides.length)
                        );
                        aGradientView.setContents(upstreamGradientViewForA);
                    }

                    if (bGradient != null) {
                        View bGradientView = new View(bGradient.getSciCoreBackend(), bGradient.getDataContainer(), new long[]{numElementsInConcatDimensionInB}, offsetInB, Arrays.copyOfRange(bStrides, superDimension + 1, bStrides.length));
                        View upstreamGradientViewForB = new View(
                                upstreamGradient.getSciCoreBackend(), upstreamGradient.getDataContainer(), new long[]{numElementsInConcatDimensionInB},
                                offsetInUpstreamGradient + numElementsInConcatDimensionInA,
                                Arrays.copyOfRange(upstreamGradientStrides, superDimension + 1, upstreamGradientStrides.length)
                        );
                        bGradientView.setContents(upstreamGradientViewForB);
                    }
                }
                if (aGradient != null) {
                    a.accumulateGradient(aGradient);
                }
                if (bGradient != null) {
                    b.accumulateGradient(bGradient);
                }
            }
        }
    }
}
