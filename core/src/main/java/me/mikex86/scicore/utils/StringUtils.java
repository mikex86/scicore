package me.mikex86.scicore.utils;

import org.jetbrains.annotations.NotNull;

import java.util.Locale;

import static me.mikex86.scicore.ITensor.EPSILON;

public class StringUtils {

    /**
     * Formats the float value to a string.
     * Shows at least one decimal place, but never more than needed. Shows up to 6 decimal places.
     * Large values are represented in scientific notation
     * @param value the float to format
     * @return the formatted string
     */
    @NotNull
    public static String formatFloat(double value) {
        if (value == 0) {
            return "0.0";
        }
        if (value < 0) {
            return "-" + formatFloat(-value);
        }
        if (value <= 1e-3 || value >= 1e6) {
            return String.format(Locale.US, "%.6e", value);
        }
        int nDecimalPlaces = 0;
        double tmpD = value;
        while (Math.abs(tmpD - Math.round(tmpD)) > EPSILON) {
            tmpD *= 10;
            nDecimalPlaces++;
        }
        nDecimalPlaces = Math.max(1, nDecimalPlaces);
        nDecimalPlaces = Math.min(6, nDecimalPlaces);
        return String.format(Locale.US, "%." + nDecimalPlaces + "f", value);
    }

}
