package me.mikex86.scicore.backend.impl.cuda.codegen;

import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

public class BroadcastingElementWiseOperationKernelCodeGenerator {

    @Nullable
    private final String operator;

    @Nullable
    private final String operation;

    private final long @NotNull [] shapeA;

    private final long @NotNull [] shapeB;

    private final long @NotNull [] resultShape;

    private final long @NotNull [] stridesA;

    private final long @NotNull [] stridesB;


    private final long @NotNull [] resultStrides;

    private BroadcastingElementWiseOperationKernelCodeGenerator(@Nullable String operator, @Nullable String operation, long @NotNull [] shapeA, long @NotNull [] shapeB, long @NotNull [] resultShape, long @NotNull [] stridesA, long @NotNull [] stridesB, long @NotNull [] resultStrides) {
        this.operator = operator;
        this.operation = operation;
        this.shapeA = shapeA;
        this.shapeB = shapeB;
        this.resultShape = resultShape;
        this.stridesA = stridesA;
        this.stridesB = stridesB;
        this.resultStrides = resultStrides;
    }

    public static class Builder {

        @Nullable
        private String operator;

        @Nullable
        private String operation;

        private long @Nullable [] shapeA;

        private long @Nullable [] shapeB;

        private long @Nullable [] stridesA;

        private long @Nullable [] stridesB;

        private long @Nullable [] resultShape;

        private long @Nullable [] resultStrides;

        @NotNull
        public Builder operation(@NotNull String operation) {
            this.operation = operation;
            return this;
        }

        @NotNull
        public Builder operator(@NotNull String operator) {
            this.operator = operator;
            return this;
        }

        @NotNull
        public Builder shapeA(long @NotNull [] shapeA) {
            this.shapeA = shapeA;
            return this;
        }

        @NotNull
        public Builder stridesA(long @NotNull [] stridesA) {
            this.stridesA = stridesA;
            return this;
        }

        @NotNull
        public Builder shapeB(long @NotNull [] shapeB) {
            this.shapeB = shapeB;
            return this;
        }

        @NotNull
        public Builder stridesB(long @NotNull [] stridesB) {
            this.stridesB = stridesB;
            return this;
        }

        @NotNull
        public Builder resultShape(long @NotNull [] resultShape) {
            this.resultShape = resultShape;
            return this;
        }

        @NotNull
        public Builder resultStrides(long @NotNull [] resultStrides) {
            this.resultStrides = resultStrides;
            return this;
        }

        @NotNull
        public BroadcastingElementWiseOperationKernelCodeGenerator build() {
            if (operation == null && operator == null) {
                throw new IllegalStateException("Either operation or operator must be set");
            }
            if (shapeA == null) {
                throw new IllegalStateException("Shape A is not set");
            }
            if (shapeB == null) {
                throw new IllegalStateException("Shape B is not set");
            }
            if (stridesA == null) {
                throw new IllegalStateException("Strides A is not set");
            }
            if (stridesB == null) {
                throw new IllegalStateException("Strides B is not set");
            }
            if (resultShape == null) {
                throw new IllegalStateException("Result shape is not set");
            }
            if (resultStrides == null) {
                throw new IllegalStateException("Result strides is not set");
            }
            return new BroadcastingElementWiseOperationKernelCodeGenerator(operator, operation, shapeA, shapeB, resultShape, stridesA, stridesB, resultStrides);
        }
    }

    @NotNull
    public static Builder builder() {
        return new Builder();
    }

    @NotNull
    public String generateCode() {
        StringBuilder builder = new StringBuilder();
        builder.append("{\n");

        long numElementsResult = ShapeUtils.getNumElements(resultShape);

        builder.append("\tuint64_t i = blockIdx.x * blockDim.x + threadIdx.x;\n")
                .append("\tif (i >= ").append(numElementsResult).append(") { return; }\n");

        // generate cIndexDim0, cIndexDim1, cIndexDim2, ...
        {
            for (int i = 0; i < resultShape.length; i++) {
                builder.append("\tuint64_t cIndexDim").append(i).append(" = i");
                for (int j = i; j < resultShape.length - 1; j++) {
                    builder.append(" / ").append(resultShape[j + 1]);
                }
                builder.append(" % ").append(resultShape[i]).append(";\n");
            }
        }

        // build flat index for c with strides
        {
            builder.append("\tuint64_t cIdx = ");
            for (int i = 0; i < resultShape.length; i++) {
                builder.append("cIndexDim").append(i).append(" * ").append(resultStrides[i]);
                if (i < resultStrides.length - 1) {
                    builder.append(" + ");
                }
            }
            if (resultStrides.length == 0) {
                builder.append("0");
            }
            builder.append(";\n");
        }

        // derive aIndexDim0, aIndexDim1, aIndexDim2 from cIndexDim0, cIndexDim1, cIndexDim2, ...
        {
            if (shapeA.length > 0) {
                for (int i = 0; i < shapeA.length; i++) {
                    int aDim = shapeA.length - 1 - i;
                    int cDim = resultShape.length - 1 - i;
                    if (shapeA[aDim] == 1) {
                        builder.append("\tuint64_t aIndexDim").append(aDim).append(" = 0;\n");
                    } else {
                        builder.append("\tuint64_t aIndexDim").append(aDim).append(" = cIndexDim").append(cDim).append(";\n");
                    }
                }
            } else {
                builder.append("\tuint64_t aIndexDim").append(shapeA.length).append(" = 0;\n");
            }
        }

        // build flat index for a with strides
        {
            builder.append("\tuint64_t aIdx = ");
            for (int i = 0; i < stridesA.length; i++) {
                builder.append("aIndexDim").append(i).append(" * ").append(stridesA[i]);
                if (i < stridesA.length - 1) {
                    builder.append(" + ");
                }
            }
            if (stridesA.length == 0) {
                builder.append("0");
            }
            builder.append(";\n");
        }

        // derive bIndexDim0, bIndexDim1, bIndexDim2 from cIndexDim0, cIndexDim1, cIndexDim2, ...
        {
            if (shapeB.length > 0) {
                for (int i = 0; i < shapeB.length; i++) {
                    int bDim = shapeB.length - 1 - i;
                    int cDim = resultShape.length - 1 - i;
                    if (shapeB[bDim] == 1) {
                        builder.append("\tuint64_t bIndexDim").append(bDim).append(" = 0;\n");
                    } else {
                        builder.append("\tuint64_t bIndexDim").append(bDim).append(" = cIndexDim").append(cDim).append(";\n");
                    }
                }
            } else {
                builder.append("\tuint64_t bIndexDim").append(shapeB.length).append(" = 0;\n");
            }
        }

        // build flat index for b with strides
        {
            builder.append("\tuint64_t bIdx = ");
            for (int i = 0; i < stridesB.length; i++) {
                builder.append("bIndexDim").append(i).append(" * ").append(stridesB[i]);
                if (i < stridesB.length - 1) {
                    builder.append(" + ");
                }
            }
            if (stridesB.length == 0) {
                builder.append("0");
            }
            builder.append(";\n");
        }

        if (operator != null) {
            builder.append("\tout[cIdx] = a[aIdx] ").append(operator).append(" b[bIdx];\n");
        } else if (operation != null) {
            builder.append("\tout[cIdx] = ").append(operation).append("(a[aIdx], b[bIdx]);\n");
        } else {
            throw new IllegalStateException("Either operation or operator must be set");
        }
        builder.append("}");
        return builder.toString();
    }

}
