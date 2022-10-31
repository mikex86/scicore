package me.mikex86.scicore.memory;

import com.google.common.collect.MapMaker;
import me.mikex86.scicore.utils.dispose.IDisposable;
import org.jetbrains.annotations.NotNull;

import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.util.Map;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public abstract class AbstractMemoryManager<T extends IMemoryHandle<T>> implements IMemoryManager<T> {

    @NotNull
    private final Map<T, MemoryHandleFinalizer<T>> handleToFinalizerMap = new MapMaker()
            .weakKeys()
            .makeMap();

    @NotNull
    private final ReferenceQueue<T> memoryHandleReferenceQueue = new ReferenceQueue<>();

    @NotNull
    private final ReadWriteLock readWriteLock = new ReentrantReadWriteLock();

    @NotNull
    private final Thread finalizerThread = new Thread(() -> {
        while (true) {
            Reference<? extends T> reference;
            while (true) {
                this.readWriteLock.writeLock().lock();
                try {
                    reference = memoryHandleReferenceQueue.poll();
                    if (reference == null) {
                        break;
                    }
                    if (reference instanceof MemoryHandleFinalizer<?> finalizer) {
                        finalizer.dispose();
                    } else {
                        throw new IllegalStateException("Unknown reference type: " + reference.getClass().getName());
                    }
                } finally {
                    this.readWriteLock.writeLock().unlock();
                }
            }
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                return;
            }
        }
    }, this.getClass().getSimpleName() + "-Finalizer");

    {
        finalizerThread.setDaemon(true);
        finalizerThread.start();
    }

    protected void registerFinalizer(@NotNull T memoryHandle) {
        this.handleToFinalizerMap.put(memoryHandle, new MemoryHandleFinalizer<>(memoryHandle, memoryHandleReferenceQueue, createDisposableFor(memoryHandle)));
    }

    @NotNull
    protected abstract IDisposable createDisposableFor(@NotNull T memoryHandle);

    protected void deactivateFinalizerFor(@NotNull T directMemoryHandle) {
        MemoryHandleFinalizer<T> finalizer = this.handleToFinalizerMap.remove(directMemoryHandle);
        if (finalizer != null) {
            finalizer.deactivate();
        }
    }

}
