package me.mikex86.scicore.graph.op;

import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

public interface IBiParametricOperation<F, S> extends IParametricOperation, ITrinaryOperation {

    @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor tensor, @Nullable F f, @Nullable S s);

    @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor tensor, @Nullable F f, @Nullable S s);

    @Override
    @NotNull
    default ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b, @NotNull ITensor c) {
        Validator.assertTrue(b.isScalar(), "input1 of binary operation must be a scalar.");
        Validator.assertTrue(c.isScalar(), "input2 of binary operation must be a scalar.");

        Class<F> fClass = getFirstType();
        Class<S> sClass = getSecondType();

        Validator.assertTrue(b.getDataType().isSameType(fClass), "input0 of binary operation must be of type " + fClass.getSimpleName() + ".");
        Validator.assertTrue(c.getDataType().isSameType(sClass), "input1 of binary operation must be of type " + sClass.getSimpleName() + ".");

        F f = b.element(fClass);
        S s = c.element(sClass);

        return perform(ctx, a, f, s);
    }

    @Override
    @NotNull
    default ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b, @NotNull ITensor c) {
        Validator.assertTrue(b.isScalar(), "input1 of binary operation must be a scalar.");
        Validator.assertTrue(c.isScalar(), "input2 of binary operation must be a scalar.");

        Class<F> fClass = getFirstType();
        Class<S> sClass = getSecondType();

        Validator.assertTrue(b.getDataType().isSameType(fClass), "input1 of binary operation must be of type " + fClass.getSimpleName() + ".");
        Validator.assertTrue(c.getDataType().isSameType(sClass), "input2 of binary operation must be of type " + sClass.getSimpleName() + ".");

        F f = b.element(fClass);
        S s = c.element(sClass);

        return performLazily(ctx, a, f, s);
    }

    @NotNull Class<F> getFirstType();

    @NotNull Class<S> getSecondType();

}
