package me.mikex86.scicore.op;

import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public interface IUnaryOperation extends IOperation {

    @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input);

    @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor input);

    @Override
    @NotNull
    default ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull List<@NotNull ITensor> inputs) {
        Validator.assertTrue(inputs.size() == 1, "unary operation must be supplied 1 input.");
        return perform(ctx, inputs.get(0));
    }

    @Override
    @NotNull
    default ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull List<@NotNull ITensor> inputs) {
        Validator.assertTrue(inputs.size() == 1, "unary operation must be supplied 1 input.");
        return performLazily(ctx, inputs.get(0));
    }
}
