package me.mikex86.scicore.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import org.jetbrains.annotations.NotNull;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public class OptionBundle {

    @NotNull
    private static final OptionBundle EMPTY = new OptionBundle(Collections.emptyMap());

    @NotNull
    private final Map<String, ITensor> options;

    private OptionBundle(@NotNull Map<String, ITensor> options) {
        this.options = options;
    }

    @NotNull
    public static OptionBundle of(@NotNull Map<String, ITensor> inputs) {
        return new OptionBundle(inputs);
    }

    @NotNull
    public static OptionBundle empty() {
        return EMPTY;
    }

    @NotNull
    public Optional<ITensor> get(@NotNull String name) {
        return Optional.ofNullable(options.get(name));
    }

    @NotNull
    public Optional<DataType> getDataType(@NotNull String name) {
        return get(name).map(ITensor::getDataType);
    }

    @NotNull
    public Optional<Integer> getInt(@NotNull String name) {
        return get(name).map(ITensor::getInt);
    }

    @NotNull
    public Optional<Long> getLong(@NotNull String name) {
        return get(name).map(ITensor::getLong);
    }

    @NotNull
    public Optional<Float> getFloat(@NotNull String name) {
        return get(name).map(ITensor::getFloat);
    }

    @NotNull
    public Optional<Double> getDouble(@NotNull String name) {
        return get(name).map(ITensor::getDouble);
    }

    @NotNull
    public Optional<Boolean> getBoolean(@NotNull String name) {
        return get(name).map(ITensor::getBoolean);
    }

    public boolean getOrDefault(@NotNull String name, boolean defaultValue) {
        return getBoolean(name).orElse(defaultValue);
    }

    public int getOrDefault(@NotNull String name, int defaultValue) {
        return getInt(name).orElse(defaultValue);
    }

    public long getOrDefault(@NotNull String name, long defaultValue) {
        return getLong(name).orElse(defaultValue);
    }

    public float getOrDefault(@NotNull String name, float defaultValue) {
        return getFloat(name).orElse(defaultValue);
    }

    public double getOrDefault(@NotNull String name, double defaultValue) {
        return getDouble(name).orElse(defaultValue);
    }
}
