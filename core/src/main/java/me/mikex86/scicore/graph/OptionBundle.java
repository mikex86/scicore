package me.mikex86.scicore.graph;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.backend.ISciCoreBackend;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

public class OptionBundle {

    @NotNull
    private final Map<String, ITensor> options;

    @NotNull
    private final ISciCoreBackend backend;

    private OptionBundle(@NotNull ISciCoreBackend backend, @NotNull Map<String, ITensor> options) {
        this.backend = backend;
        this.options = options;
    }

    @NotNull
    public static OptionBundle of(@NotNull ISciCoreBackend backend, @NotNull Map<String, ITensor> inputs) {
        return new OptionBundle(backend, inputs);
    }

    @NotNull
    public static OptionBundle newEmpty(@NotNull ISciCoreBackend backend) {
        return new OptionBundle(backend, new HashMap<>());
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

    public void set(@NotNull String name, @NotNull ITensor value) {
        options.put(name, value);
    }

    public void setInt(@NotNull String name, int value) {
        ITensor tensor = backend.createTensor(DataType.INT32, new long[]{1});
        tensor.setIntFlat(value, 0);
        set(name, tensor);
    }

    public void setLong(@NotNull String name, long value) {
        ITensor tensor = backend.createTensor(DataType.INT64, new long[]{1});
        tensor.setLongFlat(value, 0);
        set(name, tensor);
    }

    public void setFloat(@NotNull String name, float value) {
        ITensor tensor = backend.createTensor(DataType.FLOAT32, new long[]{1});
        tensor.setFloatFlat(value, 0);
        set(name, tensor);
    }

    public void setDouble(@NotNull String name, double value) {
        ITensor tensor = backend.createTensor(DataType.FLOAT64, new long[]{1});
        tensor.setDoubleFlat(value, 0);
        set(name, tensor);
    }

    public void setBoolean(@NotNull String name, boolean value) {
        ITensor tensor = backend.createTensor(DataType.BOOLEAN, new long[]{1});
        tensor.setBooleanFlat(value, 0);
        set(name, tensor);
    }

}
