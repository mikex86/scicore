package me.mikex86.scicore.backend.impl.cuda;

import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;

import static org.lwjgl.cuda.CU.cuGetErrorName;

public class Validator {

    public static void cuCheck(int result) {
        if (result != 0) {
            String errorName;
            try (MemoryStack stack = MemoryStack.stackPush()) {
                PointerBuffer buffer = stack.mallocPointer(1);
                cuGetErrorName(result, buffer);
                errorName = buffer.getStringUTF8();
            }
            throw new RuntimeException("CUDA error: " + errorName);
        }
    }

}
