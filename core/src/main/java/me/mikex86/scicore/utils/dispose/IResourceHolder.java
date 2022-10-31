package me.mikex86.scicore.utils.dispose;

import org.jetbrains.annotations.NotNull;

public interface IResourceHolder {

    @NotNull IDisposable getDisposableResources();

}
