package me.mikex86.scicore.backend.impl.cuda.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.LazyTensor;
import me.mikex86.scicore.View;
import me.mikex86.scicore.backend.impl.cuda.CudaBackend;
import me.mikex86.scicore.backend.impl.cuda.CudaTensor;
import me.mikex86.scicore.backend.impl.cuda.memory.MemoryHandle;
import me.mikex86.scicore.op.Graph;
import me.mikex86.scicore.op.IDifferentiableBinaryOperation;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.utils.Validator;
import me.mikex86.scicore.utils.ViewUtils;
import org.jetbrains.annotations.NotNull;

import static jcuda.jcublas.JCublas.cublasSgemm;

public class CudaMatmulOp implements IDifferentiableBinaryOperation {

    @NotNull
    private final CudaBackend backend;

    public CudaMatmulOp(@NotNull CudaBackend backend) {
        this.backend = backend;
    }

    @Override
    public @NotNull ITensor perform(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shape = a.getShape();
        long[] otherShape = b.getShape();
        Validator.assertTrue(otherShape.length == 2, "Only 2D matrices are supported");
        Validator.assertTrue(shape.length == 2, "Only 2D matrices are supported");
        Validator.assertTrue(shape[1] == otherShape[0], "Shape mismatch. A.shape[1] != B.shape[0]");
        Validator.assertTrue(a.getDataType().isNumeric(), "Data type of A is not numeric");
        Validator.assertTrue(b.getDataType().isNumeric(), "Data type of B is not numeric");
        long[] resultShape = new long[]{shape[0], otherShape[1]};
        DataType ownDataType = a.getDataType();
        DataType otherDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(ownDataType, otherDataType);

        CudaTensor result = new CudaTensor(this.backend, resultDataType, resultShape);
        MemoryHandle resultMemoryHandle = result.getDataContainer().getDeviceMemoryHandle();

        {
            // Swap A and B because cublasSgemm expects column-major matrices, and we have row-major.
            // Now, because B.T * A.T = C.T and A and B are already transposed when interpreted as column-major matrices, the result is C when interpreted as row-major.
            ITensor tmp = a;
            a = b;
            b = tmp;
        }

        int m = (int) shape[0]; // rows of A
        int k = (int) shape[1]; // columns of A and rows of B
        int n = (int) otherShape[1]; // columns of B


        MemoryHandle aDevicePtr, bDevicePtr;
        boolean aNeedsFree = false, bNeedsFree = false;
        {
            if (a instanceof CudaTensor aCudaTensor) {
                aDevicePtr = aCudaTensor.getDataContainer().getDeviceMemoryHandle();
            } else if (a instanceof View view && ViewUtils.getViewed(view) instanceof CudaTensor aCudaTensor) {
                long offset = aCudaTensor.getDataType().getSizeOf(ViewUtils.getTotalOffset(view));
                aDevicePtr = aCudaTensor.getDataContainer().getDeviceMemoryHandle().offset(offset);
            } else {
                aDevicePtr = backend.getMemoryManager().copyToDevice(a);
                aNeedsFree = true;
            }

            if (b instanceof CudaTensor bCudaTensor) {
                bDevicePtr = bCudaTensor.getDataContainer().getDeviceMemoryHandle();
            } else if (b instanceof View view && ViewUtils.getViewed(view) instanceof CudaTensor bCudaTensor) {
                long offset = ViewUtils.getTotalOffset(view);
                bDevicePtr = bCudaTensor.getDataContainer().getDeviceMemoryHandle().offset(offset);
            } else {
                bDevicePtr = backend.getMemoryManager().copyToDevice(b);
                bNeedsFree = true;
            }
        }

        cublasSgemm(
                'n', 'n',
                m, n, k,
                1f,
                aDevicePtr.getDevicePtr(),
                m,
                bDevicePtr.getDevicePtr(),
                k,
                0f,
                resultMemoryHandle.getDevicePtr(),
                m
        );

        if (aNeedsFree) {
            aDevicePtr.free();
        }

        if (bNeedsFree) {
            bDevicePtr.free();
        }

        return result;
    }

    @Override
    public @NotNull ITensor performLazily(@NotNull Graph.IOperationContext ctx, @NotNull ITensor a, @NotNull ITensor b) {
        long[] shape = a.getShape();
        long[] otherShape = b.getShape();
        Validator.assertTrue(otherShape.length == 2, "Only 2D matrices are supported");
        Validator.assertTrue(shape.length == 2, "Only 2D matrices are supported");
        Validator.assertTrue(shape[1] == otherShape[0], "Shape mismatch. A.shape[1] != B.shape[0]");
        Validator.assertTrue(a.getDataType().isNumeric(), "Data type of A is not numeric");
        Validator.assertTrue(b.getDataType().isNumeric(), "Data type of B is not numeric");
        long[] resultShape = new long[]{shape[0], otherShape[1]};
        DataType ownDataType = a.getDataType();
        DataType otherDataType = b.getDataType();
        DataType resultDataType = DataType.getLarger(ownDataType, otherDataType);
        return new LazyTensor(this.backend, resultShape, resultDataType, () -> perform(ctx, a, b));
    }

    @Override
    public void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient a, @NotNull IGraph.ITensorNodeWithGradient b) {

    }

}
