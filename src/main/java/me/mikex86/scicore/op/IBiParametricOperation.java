package me.mikex86.scicore.op;

import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

public interface IBiParametricOperation<F, S> extends IParametricOperation, ITrinaryOperation {

    @NotNull ITensor perform(@NotNull ITensor tensor, @Nullable F f, @Nullable S s);

    @NotNull ITensor performLazily(@NotNull ITensor tensor, @Nullable F f, @Nullable S s);

    @Override
    @NotNull
    default ITensor perform(@NotNull ITensor a, @NotNull ITensor b, @NotNull ITensor c) {
        Validator.assertTrue(b.isScalar(), "input1 of binary operation must be a scalar.");
        Validator.assertTrue(c.isScalar(), "input2 of binary operation must be a scalar.");

        Class<F> fClass = getFirstType();
        Class<S> sClass = getSecondType();

        Validator.assertTrue(b.getDataType().isSameType(fClass), "input0 of binary operation must be of type " + fClass.getSimpleName() + ".");
        Validator.assertTrue(c.getDataType().isSameType(sClass), "input1 of binary operation must be of type " + sClass.getSimpleName() + ".");

        F f = b.element(fClass);
        S s = c.element(sClass);

        return perform(a, f, s);
    }

    @Override
    @NotNull
    default ITensor performLazily(@NotNull ITensor a, @NotNull ITensor b, @NotNull ITensor c) {
        Validator.assertTrue(b.isScalar(), "input1 of binary operation must be a scalar.");
        Validator.assertTrue(c.isScalar(), "input2 of binary operation must be a scalar.");

        Class<F> fClass = getFirstType();
        Class<S> sClass = getSecondType();

        Validator.assertTrue(b.getDataType().isSameType(fClass), "input1 of binary operation must be of type " + fClass.getSimpleName() + ".");
        Validator.assertTrue(c.getDataType().isSameType(sClass), "input2 of binary operation must be of type " + sClass.getSimpleName() + ".");

        F f = b.element(fClass);
        S s = c.element(sClass);

        return performLazily(a, f, s);
    }

    @NotNull Class<F> getFirstType();

    @NotNull Class<S> getSecondType();

}
