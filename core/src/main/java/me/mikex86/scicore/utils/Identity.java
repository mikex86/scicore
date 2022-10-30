package me.mikex86.scicore.utils;

import org.jetbrains.annotations.NotNull;

/**
 * Wrapper for objects with equals and hashCode methods that should be ignored in favor of identity comparison.
 * @param value The value to wrap.
 * @param <T> type of the value
 */
public record Identity<T>(T value) {

    @Override
    public int hashCode() {
        return System.identityHashCode(value);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Identity<?> identity = (Identity<?>) o;
        return value == identity.value;
    }

    @NotNull
    public static <T> Identity<T> of(@NotNull T value) {
        return new Identity<>(value);
    }
}
