package me.mikex86.scicore.nn;

import me.mikex86.scicore.tensor.ITensor;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public interface IModule {

    @NotNull
    ITensor forward(@NotNull ITensor input);

    @NotNull
    List<ITensor> parameters();


    @NotNull default List<ITensor> collectParameters(@NotNull IModule... modules) {
        return Arrays.stream(modules).flatMap(m -> m.parameters().stream()).collect(Collectors.toList());
    }

}
