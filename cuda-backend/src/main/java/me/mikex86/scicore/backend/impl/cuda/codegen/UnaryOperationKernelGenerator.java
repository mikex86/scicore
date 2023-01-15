package me.mikex86.scicore.backend.impl.cuda.codegen;

import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

public class UnaryOperationKernelGenerator {

    @Nullable
    private final String operation;

    private final long @NotNull [] shape;

    private UnaryOperationKernelGenerator(@Nullable String operation, long @NotNull [] shape) {
        this.operation = operation;
        this.shape = shape;
    }

    public static class Builder {

        @Nullable
        private String operation;

        private long @Nullable [] shape;

        @NotNull
        public Builder operation(@NotNull String operation) {
            this.operation = operation;
            return this;
        }

        @NotNull
        public Builder shape(long @NotNull [] shape) {
            this.shape = shape;
            return this;
        }

        @NotNull
        public UnaryOperationKernelGenerator build() {
            if (operation == null) {
                throw new IllegalStateException("Operation must be set");
            }
            if (shape == null) {
                throw new IllegalStateException("Shape is not set");
            }
            return new UnaryOperationKernelGenerator(operation, shape);
        }

    }

    public static @NotNull Builder builder() {
        return new Builder();
    }

    @NotNull
    public String generateCode() {
        StringBuilder builder = new StringBuilder();
        builder.append("{\n");

        long numElementsResult = ShapeUtils.getNumElements(shape);

        builder.append("\tuint64_t i = blockIdx.x * blockDim.x + threadIdx.x;\n")
                .append("\tif (i >= ").append(numElementsResult).append(") { return; }\n");

        builder.append("\t").append("out[i] = ");

        if (operation != null) {
            builder.append(operation);
        }

        builder.append(";\n");
        builder.append("}\n");
        return builder.toString();
    }
}
