package me.mikex86.scicore.utils;

import me.mikex86.scicore.backend.ISciCoreBackend;
import me.mikex86.scicore.memory.DirectMemoryHandle;
import me.mikex86.scicore.tensor.DataType;
import org.jetbrains.annotations.NotNull;

import java.nio.*;

public class TensorContentUtils {

    @NotNull
    public static DirectMemoryHandle relayout(@NotNull ISciCoreBackend backend, @NotNull DirectMemoryHandle inputMemory, long[] inputShape, long[] inputStrides, long[] outputStrides, DataType dataType) {
        // TODO: MOVE TO NATIVE
        long[] index = new long[inputShape.length];
        DirectMemoryHandle contiguousMemory = backend.getDirectMemoryManager().alloc(ShapeUtils.getNumElements(inputShape), dataType);
        switch (dataType) {
            case INT8 -> {
                ByteBuffer srcBuffer = inputMemory.asByteBuffer();
                ByteBuffer dstBuffer = contiguousMemory.asByteBuffer();
                do {
                    long srcFlatIndex = ShapeUtils.getFlatIndex(index, inputShape, inputStrides);
                    long dstFlatIndex = ShapeUtils.getFlatIndex(index, inputShape, outputStrides);
                    dstBuffer.put(Math.toIntExact(dstFlatIndex), srcBuffer.get(Math.toIntExact(srcFlatIndex)));
                } while (ShapeUtils.incrementIndex(index, inputShape));
                dstBuffer.flip();
            }
            case INT16 -> {
                ShortBuffer srcBuffer = inputMemory.asShortBuffer();
                ShortBuffer dstBuffer = contiguousMemory.asShortBuffer();
                do {
                    long srcFlatIndex = ShapeUtils.getFlatIndex(index, inputShape, inputStrides);
                    long dstFlatIndex = ShapeUtils.getFlatIndex(index, inputShape, outputStrides);
                    dstBuffer.put(Math.toIntExact(dstFlatIndex), srcBuffer.get(Math.toIntExact(srcFlatIndex)));
                } while (ShapeUtils.incrementIndex(index, inputShape));
                dstBuffer.flip();
            }
            case INT32 -> {
                IntBuffer srcBuffer = inputMemory.asIntBuffer();
                IntBuffer dstBuffer = contiguousMemory.asIntBuffer();
                do {
                    long srcFlatIndex = ShapeUtils.getFlatIndex(index, inputShape, inputStrides);
                    long dstFlatIndex = ShapeUtils.getFlatIndex(index, inputShape, outputStrides);
                    dstBuffer.put(Math.toIntExact(dstFlatIndex), srcBuffer.get(Math.toIntExact(srcFlatIndex)));
                } while (ShapeUtils.incrementIndex(index, inputShape));
                dstBuffer.flip();
            }
            case INT64 -> {
                LongBuffer srcBuffer = inputMemory.asLongBuffer();
                LongBuffer dstBuffer = contiguousMemory.asLongBuffer();
                do {
                    long srcFlatIndex = ShapeUtils.getFlatIndex(index, inputShape, inputStrides);
                    long dstFlatIndex = ShapeUtils.getFlatIndex(index, inputShape, outputStrides);
                    dstBuffer.put(Math.toIntExact(dstFlatIndex), srcBuffer.get(Math.toIntExact(srcFlatIndex)));
                } while (ShapeUtils.incrementIndex(index, inputShape));
                dstBuffer.flip();
            }
            case FLOAT32 -> {
                FloatBuffer srcBuffer = inputMemory.asFloatBuffer();
                FloatBuffer dstBuffer = contiguousMemory.asFloatBuffer();
                do {
                    long srcFlatIndex = ShapeUtils.getFlatIndex(index, inputShape, inputStrides);
                    long dstFlatIndex = ShapeUtils.getFlatIndex(index, inputShape, outputStrides);
                    dstBuffer.put(Math.toIntExact(dstFlatIndex), srcBuffer.get(Math.toIntExact(srcFlatIndex)));
                } while (ShapeUtils.incrementIndex(index, inputShape));
                dstBuffer.flip();
            }
            case FLOAT64 -> {
                DoubleBuffer srcBuffer = inputMemory.asDoubleBuffer();
                DoubleBuffer dstBuffer = contiguousMemory.asDoubleBuffer();
                do {
                    long srcFlatIndex = ShapeUtils.getFlatIndex(index, inputShape, inputStrides);
                    long dstFlatIndex = ShapeUtils.getFlatIndex(index, inputShape, outputStrides);
                    dstBuffer.put(Math.toIntExact(dstFlatIndex), srcBuffer.get(Math.toIntExact(srcFlatIndex)));
                } while (ShapeUtils.incrementIndex(index, inputShape));
                dstBuffer.flip();
            }
        }
        return contiguousMemory;
    }


}
