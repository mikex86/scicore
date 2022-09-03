package me.mikex86.scicore.backend.impl.cuda.kernel;

import jcuda.Pointer;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

public record CudaKernelLaunchConfig(
        int gridDimX, int gridDimY, int gridDimZ,
        int blockDimX, int blockDimY, int blockDimZ,
        int sharedMemBytes,
        @NotNull Pointer arguments
) {

    @NotNull
    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {

        private int gridDimX = 1;
        private int gridDimY = 1;
        private int gridDimZ = 1;
        private int blockDimX = 1;
        private int blockDimY = 1;
        private int blockDimZ = 1;
        private int sharedMemBytes = 0;
        @Nullable
        private Pointer parameters = null;

        @NotNull
        public Builder gridDimX(int gridDimX) {
            this.gridDimX = gridDimX;
            return this;
        }

        @NotNull
        public Builder gridDimY(int gridDimY) {
            this.gridDimY = gridDimY;
            return this;
        }

        @NotNull
        public Builder gridDimZ(int gridDimZ) {
            this.gridDimZ = gridDimZ;
            return this;
        }

        @NotNull
        public Builder blockDimX(int blockDimX) {
            this.blockDimX = blockDimX;
            return this;
        }

        @NotNull
        public Builder blockDimY(int blockDimY) {
            this.blockDimY = blockDimY;
            return this;
        }

        @NotNull
        public Builder blockDimZ(int blockDimZ) {
            this.blockDimZ = blockDimZ;
            return this;
        }

        @NotNull
        public Builder sharedMemBytes(int sharedMemBytes) {
            this.sharedMemBytes = sharedMemBytes;
            return this;
        }

        @NotNull
        public Builder parameters(@NotNull Pointer parameters) {
            this.parameters = parameters;
            return this;
        }

        @NotNull
        public CudaKernelLaunchConfig build() {
            if (parameters == null) {
                throw new IllegalStateException("Parameters not set!");
            }
            return new CudaKernelLaunchConfig(gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, parameters);
        }
    }

}
