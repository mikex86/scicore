package me.mikex86.scicore.memory;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.lwjgl.system.MemoryUtil;

import java.nio.*;

public class DirectMemoryHandle implements IMemoryHandle<DirectMemoryHandle> {

    @NotNull
    private final DirectMemoryManager memoryManager;

    @Nullable
    private final DirectMemoryHandle parent;

    private final long nativePtr;
    private final long size;

    boolean freed = false;

    public DirectMemoryHandle(@NotNull DirectMemoryManager memoryManager, long nativePtr, long nBytes) {
        this.memoryManager = memoryManager;
        this.parent = null;
        this.nativePtr = nativePtr;
        this.size = nBytes;
    }

    private DirectMemoryHandle(@NotNull DirectMemoryManager memoryManager, @NotNull DirectMemoryHandle parent, long offset) {
        this.memoryManager = memoryManager;
        this.parent = parent;
        this.nativePtr = parent.nativePtr + offset;
        this.size = parent.size - offset;
    }

    private DirectMemoryHandle(@NotNull DirectMemoryManager memoryManager, @NotNull DirectMemoryHandle parent, long offset, long size) {
        this.memoryManager = memoryManager;
        this.parent = parent;
        this.nativePtr = parent.nativePtr + offset;
        this.size = size;
    }

    public long getNativePtr() {
        return nativePtr;
    }

    @Override
    public long getSize() {
        return size;
    }

    @Override
    public @Nullable DirectMemoryHandle getParent() {
        return parent;
    }

    @Override
    public void free() {
        if (freed) {
            return;
        }
        if (parent != null) {
            return; // parent will free
        }
        memoryManager.free(this);
        freed = true;
    }

    @Override
    public @NotNull DirectMemoryHandle offset(long offset) {
        if (offset < 0) {
            throw new IllegalArgumentException("offset must be >= 0");
        }
        if (offset >= size) {
            throw new IllegalArgumentException("offset must be < size");
        }
        return new DirectMemoryHandle(memoryManager, this, offset);
    }

    @Override
    public @NotNull DirectMemoryHandle offset(long offset, long size) {
        if (offset < 0) {
            throw new IllegalArgumentException("offset must be >= 0");
        }
        if (offset >= this.size) {
            throw new IllegalArgumentException("offset must be < size");
        }
        if (size < 0) {
            throw new IllegalArgumentException("size must be >= 0");
        }
        if (offset + size > this.size) {
            throw new IllegalArgumentException("offset + size must be <= size");
        }
        return new DirectMemoryHandle(memoryManager, this, offset, size);
    }

    @NotNull
    public DirectMemoryHandle restrictSize(long size) {
        if (size < 0) {
            throw new IllegalArgumentException("size must be >= 0");
        }
        if (size > this.size) {
            throw new IllegalArgumentException("size must be <= this.size");
        }
        return new DirectMemoryHandle(memoryManager, this, 0, size);
    }

    @Override
    @SuppressWarnings("deprecation")
    public void finalize() throws Throwable {
        super.finalize();
        free();
    }


    @Override
    public boolean isFreed() {
        return freed;
    }

    @NotNull
    public ByteBuffer asByteBuffer() {
        long size = getSize();
        if (size > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Cannot create a ByteBuffer for a memory handle larger than Integer.MAX_VALUE");
        }
        return MemoryUtil.memByteBuffer(nativePtr, Math.toIntExact(size));
    }

    @NotNull
    public ShortBuffer asShortBuffer() {
        long size = getSize() / Short.BYTES;
        if (size > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Cannot create a ShortBuffer for a memory handle larger than Integer.MAX_VALUE");
        }
        return MemoryUtil.memShortBuffer(nativePtr, Math.toIntExact(size));
    }

    @NotNull
    public IntBuffer asIntBuffer() {
        long size = getSize() / Integer.BYTES;
        if (size > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Cannot create a IntBuffer for a memory handle larger than Integer.MAX_VALUE");
        }
        return MemoryUtil.memIntBuffer(nativePtr, Math.toIntExact(size));
    }

    @NotNull
    public LongBuffer asLongBuffer() {
        long size = getSize() / Long.BYTES;
        if (size > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Cannot create a LongBuffer for a memory handle larger than Integer.MAX_VALUE");
        }
        return MemoryUtil.memLongBuffer(nativePtr, Math.toIntExact(size));
    }

    @NotNull
    public FloatBuffer asFloatBuffer() {
        long size = getSize() / Float.BYTES;
        if (size > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Cannot create a FloatBuffer for a memory handle larger than Integer.MAX_VALUE");
        }
        return MemoryUtil.memFloatBuffer(nativePtr, Math.toIntExact(size));
    }

    @NotNull
    public DoubleBuffer asDoubleBuffer() {
        long size = getSize() / Double.BYTES;
        if (size > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Cannot create a DoubleBuffer for a memory handle larger than Integer.MAX_VALUE");
        }
        return MemoryUtil.memDoubleBuffer(nativePtr, Math.toIntExact(size));
    }

    @Override
    public String toString() {
        return "DirectMemoryHandle{" +
                "parent=" + parent +
                ", nativePtr=" + nativePtr +
                ", size=" + size +
                ", freed=" + freed +
                '}';
    }
}
