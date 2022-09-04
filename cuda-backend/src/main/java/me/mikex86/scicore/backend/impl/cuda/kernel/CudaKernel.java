package me.mikex86.scicore.backend.impl.cuda.kernel;

import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import org.jetbrains.annotations.NotNull;

import java.io.InputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static jcuda.driver.JCudaDriver.*;
import static me.mikex86.scicore.backend.impl.cuda.Validator.cuCheck;

public class CudaKernel {

    @NotNull
    private final CUmodule cuModule;

    @NotNull
    private final Map<String, CUfunction> cuFunctions = new HashMap<>();

    private boolean freed = false;

    private CudaKernel(@NotNull String ptxCode, @NotNull List<String> functionNames) {
        this.cuModule = new CUmodule();
        cuCheck(cuModuleLoadData(cuModule, ptxCode));
        for (String functionName : functionNames) {
            CUfunction cuFunction = new CUfunction();
            cuModuleGetFunction(cuFunction, cuModule, functionName);
            cuFunctions.put(functionName, cuFunction);
        }
    }

    @NotNull
    public static CudaKernel load(@NotNull String ptxCode, @NotNull List<String> functionNames) {
        return new CudaKernel(ptxCode, functionNames);
    }

    @NotNull
    public static CudaKernel loadClassPath(@NotNull String resourceName, @NotNull List<String> functionNames) {
        try (InputStream inputStream = CudaKernel.class.getClassLoader().getResourceAsStream(resourceName)) {
            if (inputStream == null) {
                throw new IllegalArgumentException("Resource not found: " + resourceName);
            }
            byte[] bytes = inputStream.readAllBytes();
            String ptxCode = new String(bytes);
            return load(ptxCode, functionNames);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @NotNull
    private Optional<CUfunction> getFunction(@NotNull String functionName) {
        return Optional.ofNullable(cuFunctions.get(functionName));
    }

    public void launch(@NotNull String functionName, @NotNull CudaKernelLaunchConfig config) {
        Optional<CUfunction> functionOpt = getFunction(functionName);
        if (functionOpt.isEmpty()) {
            throw new IllegalArgumentException("Function not found: " + functionName);
        }
        launch(functionOpt.get(), config);
    }

    private void launch(@NotNull CUfunction function, @NotNull CudaKernelLaunchConfig config) {
        if (freed) {
            throw new IllegalStateException("Cuda kernel already freed!");
        }
        cuCheck(
                cuLaunchKernel(
                        function,
                        config.gridDimX(), config.gridDimY(), config.gridDimZ(),
                        config.blockDimX(), config.blockDimY(), config.blockDimZ(),
                        config.sharedMemBytes(), null, config.arguments(), null
                )
        );
    }

    // TODO: IMPLEMENT LAUNCH BLOCKING AND USE IT EVERYWHERE INSTEAD OF launch()

    public void free() {
        if (freed) {
            return;
        }
        cuCheck(cuModuleUnload(cuModule));
        freed = true;
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        free();
    }
}
