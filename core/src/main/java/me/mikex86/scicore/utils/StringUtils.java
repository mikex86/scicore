package me.mikex86.scicore.utils;

import org.jetbrains.annotations.NotNull;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Locale;

public class StringUtils {

    @NotNull
    private static final DecimalFormatSymbols EXPLICIT_POSITIVE_EXPONENT_DECIMAL_SYMBOLS = DecimalFormatSymbols.getInstance(Locale.US);


    static {
        EXPLICIT_POSITIVE_EXPONENT_DECIMAL_SYMBOLS.setExponentSeparator("e+");
    }

    @NotNull
    private static final DecimalFormat EXPLICIT_POSITIVE_EXPONENT_SCIENTIFIC_FORMAT = new DecimalFormat("0.###E0", EXPLICIT_POSITIVE_EXPONENT_DECIMAL_SYMBOLS);


    @NotNull
    private static final DecimalFormat SCIENTIFIC_FORMAT = new DecimalFormat("0.###E0");

    /**
     * Formats the float value to string of the fixed length 8.
     * Shows at least one decimal place, but never more than needed.
     * Large values are represented in scientific notation
     *
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
            if (value > 1) {
                return EXPLICIT_POSITIVE_EXPONENT_SCIENTIFIC_FORMAT.format(value);
            } else {
                return SCIENTIFIC_FORMAT.format(value);
            }
        }
        int integerPart = (int) value;
        int nDigits = 0;
        while (integerPart > 0) {
            integerPart /= 10;
            nDigits++;
        }
        int nDecimalPlaces = Math.max(0, 8 - nDigits - 2);
        return String.format(Locale.US, "%." + nDecimalPlaces + "f", value);
    }

}
