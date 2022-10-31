package me.mikex86.scicore.graph.op;

import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public interface ITrinaryOperation extends IOperation {

    @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b, @NotNull ITensor c);

    @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b, @NotNull ITensor c);

    @Override
    @NotNull default ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull List<@NotNull ITensor> inputs) {
        Validator.assertTrue(inputs.size() == 3, "trinary operation must be supplied 3 inputs.");
        return perform(ctx, inputs.get(0), inputs.get(1), inputs.get(2));
    }

    @Override
    @NotNull default ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull List<@NotNull ITensor> inputs) {
        Validator.assertTrue(inputs.size() == 3, "trinary operation must be supplied 3 inputs.");
        return performLazily(ctx, inputs.get(0), inputs.get(1), inputs.get(2));
    }
}
