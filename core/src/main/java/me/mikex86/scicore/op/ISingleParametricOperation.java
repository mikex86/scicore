package me.mikex86.scicore.op;

import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

public interface ISingleParametricOperation<T> extends IParametricOperation, IBinaryOperation {

    @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor tensor, @Nullable T t);

    @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor tensor, @Nullable T t);

    @Override
    @NotNull
    default ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        Validator.assertTrue(b.isScalar(), "input of binary operation must be a scalar.");
        Validator.assertTrue(b.getDataType().isSameType(getType()), "input of binary operation must be of type " + getType().getSimpleName() + ".");
        T t = b.element(getType());
        return perform(ctx, a, t);
    }

    @Override
    @NotNull
    default ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        Validator.assertTrue(b.isScalar(), "input of binary operation must be a scalar.");
        Validator.assertTrue(b.getDataType().isSameType(getType()), "input of binary operation must be of type " + getType().getSimpleName() + ".");
        T t = b.element(getType());
        return performLazily(ctx, a, t);
    }

    @NotNull Class<T> getType();
}
