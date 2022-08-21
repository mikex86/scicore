package me.mikex86.scicore.op;

import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public interface ITrinaryOperation extends IOperation {

    @NotNull ITensor perform(@NotNull ITensor a, @NotNull ITensor b, @NotNull ITensor c);

    @NotNull ITensor performLazily(@NotNull ITensor a, @NotNull ITensor b, @NotNull ITensor c);

    @Override
    @NotNull default ITensor perform(@NotNull List<@NotNull ITensor> inputs) {
        Validator.assertTrue(inputs.size() == 3, "trinary operation must be supplied 3 inputs.");
        return perform(inputs.get(0), inputs.get(1), inputs.get(2));
    }

    @Override
    @NotNull default ITensor performLazily(@NotNull List<@NotNull ITensor> inputs) {
        Validator.assertTrue(inputs.size() == 3, "trinary operation must be supplied 3 inputs.");
        return performLazily(inputs.get(0), inputs.get(1), inputs.get(2));
    }
}
