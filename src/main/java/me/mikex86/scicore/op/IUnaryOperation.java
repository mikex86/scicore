package me.mikex86.scicore.op;

import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public interface IUnaryOperation extends IOperation {

    @NotNull ITensor perform(@NotNull ITensor input);

    @NotNull ITensor performLazily(@NotNull ITensor input);

    @Override
    @NotNull default ITensor perform(@NotNull List<@NotNull ITensor> inputs) {
        Validator.assertTrue(inputs.size() == 1, "unary operation must be supplied 1 input.");
        return perform(inputs.get(0));
    }

    @Override
    @NotNull default ITensor performLazily(@NotNull List<@NotNull ITensor> inputs) {
        Validator.assertTrue(inputs.size() == 1, "unary operation must be supplied 1 input.");
        return performLazily(inputs.get(0));
    }
}
