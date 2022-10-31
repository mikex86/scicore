package me.mikex86.scicore.graph.op;

import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public interface IBinaryOperation extends IOperation {

    @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b);

    @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b);

    @Override
    @NotNull
    default ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull List<@NotNull ITensor> inputs) {
        Validator.assertTrue(inputs.size() == 2, "binary operation must be supplied 2 inputs.");
        return perform(ctx, inputs.get(0), inputs.get(1));
    }

    @Override
    @NotNull
    default ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull List<@NotNull ITensor> inputs) {
        Validator.assertTrue(inputs.size() == 2, "binary operation must be supplied 2 inputs.");
        return performLazily(ctx, inputs.get(0), inputs.get(1));
    }

}
