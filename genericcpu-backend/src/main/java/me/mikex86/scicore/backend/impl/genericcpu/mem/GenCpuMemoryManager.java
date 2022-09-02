package me.mikex86.scicore.backend.impl.genericcpu.mem;

import me.mikex86.scicore.DataType;
import org.jetbrains.annotations.NotNull;
import org.lwjgl.system.jemalloc.JEmalloc;

import java.nio.*;

public class GenCpuMemoryManager {

    public static final long NULL = 0;

    public long alloc(long nBytes) {
        long ptr = JEmalloc.nje_malloc(nBytes);
        if (ptr == NULL) {
            throw new OutOfMemoryError("JEmalloc failed to allocate " + nBytes + " bytes");
        }
        return ptr;
    }

    public long alloc(long nElements, @NotNull DataType dataType) {
        long nBytes = dataType.getSizeOf(nElements);
        return alloc(nBytes);
    }

    @NotNull
    public ByteBuffer allocBuffer(long nBytes) {
        ByteBuffer buffer = JEmalloc.je_malloc(nBytes);
        if (buffer == null) {
            throw new OutOfMemoryError("JEmalloc failed to allocate " + nBytes + " bytes");
        }
        return buffer;
    }

    @NotNull
    public ByteBuffer allocBuffer(long nElements, @NotNull DataType dataType) {
        long nBytes = dataType.getSizeOf(nElements);
        return allocBuffer(nBytes);
    }

    @NotNull
    public ShortBuffer allocShortBuffer(long nElements) {
        long nBytes = nElements * Short.BYTES;
        ByteBuffer buffer = JEmalloc.je_malloc(nBytes);
        if (buffer == null) {
            throw new OutOfMemoryError("JEmalloc failed to allocate " + nBytes + " bytes");
        }
        return buffer.asShortBuffer();
    }

    @NotNull
    public IntBuffer allocIntBuffer(long nElements) {
        long nBytes = nElements * Integer.BYTES;
        ByteBuffer buffer = JEmalloc.je_malloc(nBytes);
        if (buffer == null) {
            throw new OutOfMemoryError("JEmalloc failed to allocate " + nBytes + " bytes");
        }
        return buffer.asIntBuffer();
    }

    @NotNull
    public LongBuffer allocLongBuffer(long nElements) {
        long nBytes = nElements * Long.BYTES;
        ByteBuffer buffer = JEmalloc.je_malloc(nBytes);
        if (buffer == null) {
            throw new OutOfMemoryError("JEmalloc failed to allocate " + nBytes + " bytes");
        }
        return buffer.asLongBuffer();
    }

    @NotNull
    public FloatBuffer allocFloatBuffer(long nElements) {
        long nBytes = nElements * Float.BYTES;
        ByteBuffer buffer = JEmalloc.je_malloc(nBytes);
        if (buffer == null) {
            throw new OutOfMemoryError("JEmalloc failed to allocate " + nBytes + " bytes");
        }
        return buffer.asFloatBuffer();
    }

    @NotNull
    public DoubleBuffer allocDoubleBuffer(long nElements) {
        long nBytes = nElements * Double.BYTES;
        ByteBuffer buffer = JEmalloc.je_malloc(nBytes);
        if (buffer == null) {
            throw new OutOfMemoryError("JEmalloc failed to allocate " + nBytes + " bytes");
        }
        return buffer.asDoubleBuffer();
    }

    public void free(long ptr) {
        JEmalloc.nje_free(ptr);
    }

    public void free(@NotNull ByteBuffer buffer) {
        JEmalloc.je_free(buffer);
    }

    public void free(@NotNull ShortBuffer buffer) {
        JEmalloc.je_free(buffer);
    }

    public void free(@NotNull IntBuffer buffer) {
        JEmalloc.je_free(buffer);
    }

    public void free(@NotNull LongBuffer buffer) {
        JEmalloc.je_free(buffer);
    }

    public void free(@NotNull FloatBuffer buffer) {
        JEmalloc.je_free(buffer);
    }

    public void free(@NotNull DoubleBuffer buffer) {
        JEmalloc.je_free(buffer);
    }
}
