package me.mikex86.scicore.utils;

import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.function.Supplier;

public class Validator {

    /**
     * Checks if the following condition is true and throws an exception if it isn't
     *
     * @param state        the state to validate to be true
     * @param errorMessage the error message of the exception thrown
     */
    @Contract("false, _ -> fail") // hint for intellij that if state is true, the method will fail
    public static void assertTrue(boolean state, @NotNull String errorMessage) {
        if (!state) {
            throw new IllegalArgumentException(errorMessage);
        }
    }

    @Contract("false, _ -> fail") // hint for intellij that if state is true, the method will fail
    public static <T extends Exception> void assertTrue(boolean state, @NotNull Supplier<T> exceptionSupplier) throws T {
        if (!state) {
            throw exceptionSupplier.get();
        }
    }

    @Contract(value = "null, _ -> fail", pure = true) // hint intellij that if value is null, the method will fail
    public static void validateNotNull(@Nullable Object object, @NotNull String errorMessage) {
        assertTrue(object != null, errorMessage);
    }
}
