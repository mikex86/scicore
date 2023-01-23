package me.mikex86.scicore.backend.impl.cuda.op;

import me.mikex86.scicore.backend.impl.cuda.CudaBackend;
import me.mikex86.scicore.backend.impl.cuda.kernel.CudaKernel;
import me.mikex86.scicore.backend.impl.cuda.kernel.KernelNameUtility;
import me.mikex86.scicore.backend.impl.cuda.memory.CudaMemoryHandle;
import me.mikex86.scicore.backend.impl.jvm.JvmBackend;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.op.IBinaryOperation;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.LazyTensor;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public class CudaLessThanOp implements IBinaryOperation {

    @NotNull
    private final CudaBackend backend;

    @NotNull
    private final CudaKernel kernel = CudaKernel.loadClassPath("kernels/cuda/less_than.ptx", KernelNameUtility.getForAllDataTypes("less_than", List.of(DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64, DataType.FLOAT32, DataType.FLOAT64)));

    public CudaLessThanOp(@NotNull CudaBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(Graph.@NotNull IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        if (!a.getDataType().isNumeric()) {
            throw new IllegalArgumentException("A is not numeric");
        }
        if (!b.isScalar() || !b.getDataType().isNumeric()) {
            throw new IllegalArgumentException("B must be a numeric scalar");
        }
        ITensor result = backend.createTensor(DataType.BOOLEAN, a.getShape());
        double bAsDouble = b.elementAsDouble();
        double bAsLong = b.elementAsLong();

        CudaMemoryHandle aHandle = backend.getCudaMemoryManager().ensureOnDevice(a);
        // TODO: LAUNCH KERNEL


        result = result.view(a.getShape(), a.getStrides());
        return result;
    }

    @Override
    public @NotNull ITensor performLazily(Graph.@NotNull IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        if (!a.getDataType().isNumeric()) {
            throw new IllegalArgumentException("A is not numeric");
        }
        if (!b.isScalar() || !b.getDataType().isNumeric()) {
            throw new IllegalArgumentException("B must be a numeric scalar");
        }
        return new LazyTensor(backend, a.getShape(), DataType.BOOLEAN);
    }

}
