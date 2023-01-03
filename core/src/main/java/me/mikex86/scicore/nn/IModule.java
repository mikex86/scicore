package me.mikex86.scicore.nn;

import me.mikex86.scicore.nn.saveformat.ModuleSerializer;
import me.mikex86.scicore.tensor.ITensor;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public interface IModule {

    @NotNull
    ITensor forward(@NotNull ITensor input);

    @NotNull
    default ITensor invoke(@NotNull ITensor input) {
        return forward(input);
    }

    @NotNull
    default List<ITensor> parameters() {
        return subModules()
                .stream()
                .flatMap(m -> m.parameters().stream())
                .collect(Collectors.toList());
    }

    @NotNull
    default List<IModule> subModules() {
        return Collections.emptyList();
    }

    /**
     * Saves the module to a file in SciCore's save-format.
     *
     * @param path The path to save the module to.
     * @throws IOException If an I/O error occurs.
     */
    default void save(@NotNull Path path) throws IOException {
        ModuleSerializer moduleSerializer = new ModuleSerializer();
        try (OutputStream outputStream = Files.newOutputStream(path)) {
            moduleSerializer.save(this, outputStream);
        }
    }

    /**
     * Loads the model parameters from a file in SciCore's save-format.
     * The module must be of the same type as the one that was saved.
     *
     * @param path The path to load the module from.
     * @throws IOException If an I/O error occurs.
     */
    default void load(@NotNull Path path) throws IOException {
        ModuleSerializer moduleSerializer = new ModuleSerializer();
        try (InputStream inputStream = Files.newInputStream(path)) {
            moduleSerializer.load(this, inputStream);
        }
    }
}
