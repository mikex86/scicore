package me.mikex86.scicore.memory;

import me.mikex86.scicore.utils.dispose.IDisposable;
import org.jetbrains.annotations.NotNull;

import java.lang.ref.PhantomReference;
import java.lang.ref.ReferenceQueue;

public class MemoryHandleFinalizer<T extends IMemoryHandle<T>> extends PhantomReference<T> implements IDisposable {

    @NotNull
    private final IDisposable resource;

    private boolean active = true;

    public MemoryHandleFinalizer(@NotNull T referent, @NotNull ReferenceQueue<T> referenceQueue, @NotNull IDisposable resource) {
        super(referent, referenceQueue);
        this.resource = resource;
    }

    public void deactivate() {
        this.active = false;
    }

    @Override
    public void dispose() {
        if (!active) {
            return;
        }
        resource.dispose();
    }
}
