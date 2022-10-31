package me.mikex86.scicore.memory;

import me.mikex86.scicore.tensor.DataType;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

public interface IMemoryHandle<T extends IMemoryHandle<T>> {

    /**
     * @return the parent of this memory handle, or null if this is the root. Only the root handle can be freed.
     */
    @Nullable
    T getParent();

    /**
     * @return true, if this handle is responsible for freeing the memory. Only the root handle can be freed.
     */
    default boolean canFree() {
        return getParent() == null;
    }

    /**
     * Free the memory associated with this handle. Only the root handle can be freed.
     */
    void free();

    /**
     * @return the size of the memory associated with this handle.
     */
    long getSize();

    /**
     * @return a reference to this handle. The reference handle cannot be freed, as the parent handle will be responsible
     */
    @NotNull
    default T createReference() {
        return offset(0);
    }

    /**
     * Creates a reference handle to a subregion of this handle. The parent handle will be responsible for freeing the memory.
     *
     * @param offset the offset from the start of this handle
     * @return a reference handle to the subregion
     */
    @NotNull T offset(long offset);

    /**
     * Creates a reference handle to a subregion of this handle. The parent handle will be responsible for freeing the memory.
     *
     * @param offset the offset from the start of this handle
     * @param size   the size of the subregion
     * @return a reference handle to the subregion
     */
    @NotNull T offset(long offset, long size);

    /**
     * Creates a reference handle to a subregion of this handle. The parent handle will be responsible for freeing the memory.
     * @param offset the offset from the start of this handle in number of elements of the specified data type
     * @param size the size of the subregion in number of elements of the specified data type
     * @param dataType the specified data type
     * @return a reference handle to the subregion
     */
    @NotNull default T offset(long offset, long size, @NotNull DataType dataType) {
        return offset(dataType.getSizeOf(offset), dataType.getSizeOf(size));
    }

    boolean isFreed();
}
