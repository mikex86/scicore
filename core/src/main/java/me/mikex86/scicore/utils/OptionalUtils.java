package me.mikex86.scicore.utils;

import org.jetbrains.annotations.NotNull;

import java.util.Optional;

public class OptionalUtils {

    @SuppressWarnings({"unchecked", "OptionalUsedAsFieldOrParameterType"})
    public static @NotNull <I, O extends I> Optional<O> cast(@NotNull Optional<I> optional, @NotNull Class<O> clazz) {
        if (optional.isPresent()) {
            if (clazz.isAssignableFrom(optional.get().getClass())) {
                return (Optional<O>) optional;
            } else {
                throw new ClassCastException("Cannot cast " + optional.get().getClass() + " to " + clazz);
            }
        } else {
            return Optional.empty();
        }
    }

}
