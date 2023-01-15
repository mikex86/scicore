package me.mikex86.scicore.backend.impl.cuda.codegen;

import me.mikex86.scicore.tensor.DataType;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.ArrayList;
import java.util.List;

public class KernelCodeGenerator {

    @NotNull
    public static KernelCodeGenerator create() {
        return new KernelCodeGenerator();
    }

    public static class KernelFunction {

        @NotNull
        private final String prefix;

        @NotNull
        private final String returnType;

        @NotNull
        private final String functionName;

        @NotNull
        private final List<Parameter> parameters;

        @NotNull
        private final String body;

        public static class Parameter {

            private final String type;
            private final String name;

            public Parameter(String type, String name) {
                this.type = type;
                this.name = name;
            }

            @NotNull
            public String getName() {
                return name;
            }

            @NotNull
            public String getType() {
                return type;
            }

        }

        private KernelFunction(@NotNull String prefix, @NotNull String returnType, @NotNull String functionName, @NotNull List<Parameter> parameters, @NotNull String body) {
            this.prefix = prefix;
            this.returnType = returnType;
            this.functionName = functionName;
            this.parameters = parameters;
            this.body = body;
        }


        @NotNull
        public String buildCode() {
            StringBuilder sb = new StringBuilder();
            sb.append(prefix).append(" ").append(returnType).append(" ").append(functionName).append("(");
            for (int i = 0; i < parameters.size(); i++) {
                Parameter parameter = parameters.get(i);
                sb.append(parameter.getType()).append(" ").append(parameter.getName());
                if (i < parameters.size() - 1) {
                    sb.append(", ");
                }
            }
            sb.append(") ");
            sb.append(body);
            return sb.toString();
        }

        public static class Builder {

            @Nullable
            private String prefix;

            @Nullable
            private String returnType;

            @Nullable
            private String functionName;

            @NotNull
            private final List<Parameter> parameters = new ArrayList<>();

            @Nullable
            private String body;

            @NotNull
            public Builder prefix(@NotNull String prefix) {
                this.prefix = prefix;
                return this;
            }

            @NotNull
            public Builder returnType(@NotNull String returnType) {
                this.returnType = returnType;
                return this;
            }

            @NotNull
            public Builder functionName(@NotNull String functionName) {
                this.functionName = functionName;
                return this;
            }

            @NotNull
            public Builder parameters(@NotNull List<Parameter> parameters) {
                this.parameters.addAll(parameters);
                return this;
            }

            @NotNull
            public Builder parameter(@NotNull String type, @NotNull String name) {
                this.parameters.add(new Parameter(type, name));
                return this;
            }

            @NotNull
            public Builder parameter(@NotNull DataType type, int pointerLevel, @NotNull String name) {
                this.parameters.add(new Parameter(DataTypeUtils.getCudaType(type) + "*".repeat(pointerLevel), name));
                return this;
            }

            @NotNull
            public Builder body(@NotNull String body) {
                this.body = body;
                return this;
            }


            @NotNull
            public KernelFunction build() {
                if (prefix == null) {
                    throw new IllegalStateException("Prefix not set");
                }
                if (returnType == null) {
                    throw new IllegalStateException("returnType is not set");
                }
                if (functionName == null) {
                    throw new IllegalStateException("functionName is not set");
                }
                if (body == null) {
                    throw new IllegalStateException("body is not set");
                }
                return new KernelFunction(prefix, returnType, functionName, parameters, body);
            }

        }

        @NotNull
        public static Builder builder() {
            return new Builder();
        }

    }

    @NotNull
    private final List<KernelFunction> kernelFunctions = new ArrayList<>();

    @NotNull
    public KernelCodeGenerator addFunction(@NotNull KernelFunction kernelFunction) {
        kernelFunctions.add(kernelFunction);
        return this;
    }

    private static final String KERNEL_CODE_PREFIX = "typedef char int8_t;\n" +
                                                     "typedef unsigned char uint8_t;\n" +
                                                     "typedef short int16_t;\n" +
                                                     "typedef unsigned short uint16_t;\n" +
                                                     "typedef int int32_t;\n" +
                                                     "typedef unsigned int uint32_t;\n" +
                                                     "typedef long long int int64_t;\n" +
                                                     "typedef unsigned long long int uint64_t;\n\n";

    @NotNull
    public String buildCode() {
        StringBuilder sb = new StringBuilder(KERNEL_CODE_PREFIX).append(" ");
        for (KernelFunction kernelFunction : kernelFunctions) {
            sb.append(kernelFunction.buildCode()).append('\n');
        }
        return sb.toString();
    }

}
