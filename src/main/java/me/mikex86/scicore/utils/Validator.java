package me.mikex86.scicore.utils;

import org.jetbrains.annotations.NotNull;

public class Validator {

    /**
     * Checks if the following condition is true and throws an exception if it isn't
     *
     * @param state        the state to validate to be true
     * @param errorMessage the error message of the exception thrown
     */
    public static void assertTrue(boolean state, @NotNull String errorMessage) {
        if (!state) {
            throw new IllegalStateException(errorMessage);
        }
    }
}
