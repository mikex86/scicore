package me.mikex86.scicore.memory;

import org.jetbrains.annotations.NotNull;

/**
 * Utility to use java syntax sugar to automatically release a memory handle.
 * Freeing memory handles is optional and not required, but it is recommended to do so, if it can be determined early that the handle is no longer needed.
 * The utility is built with this in mind. The utility will not throw an exception, if the handle is a reference to another handle and thus cannot be freed.
 * In such a case, the guard will simply do nothing. The guard will however throw an exception, if the handle is already freed.
 * @param <T> The type of the handle.
 */
public class MemoryHandleGuard<T extends IMemoryHandle<T>> implements AutoCloseable {

    @NotNull
    private final T memoryHandle;

    private MemoryHandleGuard(@NotNull T memoryHandle) {
        this.memoryHandle = memoryHandle;
    }

    @NotNull
    public static <T extends IMemoryHandle<T>> MemoryHandleGuard<T> guard(@NotNull T memoryHandle) {
        return new MemoryHandleGuard<>(memoryHandle);
    }

    @NotNull
    public T getGuardedHandle() {
        return memoryHandle;
    }

    @Override
    public void close() {
        if (memoryHandle.isFreed()) {
            throw new IllegalArgumentException("Memory handle is already freed");
        }
        if (!memoryHandle.canFree()) {
            return;
        }
        memoryHandle.free();
    }
}
