package me.mikex86.scicore.utils;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class StringUtilsTest {

    @Test
    void formatFloat_0() {
        assertEquals("0.0", StringUtils.formatFloat(0));
    }

    @Test
    void formatFloat_showsAtLeastOneDecimalPlace() {
        assertEquals("14.0", StringUtils.formatFloat(14));
    }

    @Test
    void formatFloat_negative() {
        assertEquals("-16.0", StringUtils.formatFloat(-16));
    }

    @Test
    void formatFloat_largeValuesScientificNotation() {
        assertEquals("1.000000e+06", StringUtils.formatFloat(1e6));
    }

    @Test
    void formatFloat_largeValuesScientificNotation_2() {
        assertEquals("1.123456e+06", StringUtils.formatFloat(1.123456e6));
    }

    @Test
    void formatFloat_smallValuesScientificNotation() {
        assertEquals("1.000000e-03", StringUtils.formatFloat(1e-3));
    }

    @Test
    void formatFloat_smallValuesScientificNotation_2() {
        assertEquals("1.123456e-04", StringUtils.formatFloat(1.123456e-4));
    }

    @Test
    void formatFloat_showsUpTo6DecimalPlaces() {
        assertEquals("1.123456", StringUtils.formatFloat(1.123456));
    }

    @Test
    void formatFloat_showsUpTo6DecimalPlaces_2() {
        assertEquals("1.123457", StringUtils.formatFloat(1.1234567));
    }

    @Test
    void formatFloat_showsUpTo6DecimalPlaces_3() {
        assertEquals("1.123457", StringUtils.formatFloat(1.12345678));
    }

    @Test
    void formatFloat_showsNoMoreThanNeeded() {
        assertEquals("1.1", StringUtils.formatFloat(1.1));
    }

    @Test
    void formatFloat_showsNoMoreThanNeeded_2() {
        assertEquals("1.12", StringUtils.formatFloat(1.12));
    }

    @Test
    void formatFloat_showsNoMoreThanNeeded_3() {
        assertEquals("1.123", StringUtils.formatFloat(1.123));
    }

    @Test
    void formatFloat_showsNoMoreThanNeeded_4() {
        assertEquals("1.1234", StringUtils.formatFloat(1.1234));
    }
}