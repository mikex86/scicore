package me.mikex86.scicore.nn.saveformat;

import me.mikex86.scicore.nn.IModule;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.utils.ShapeUtils;
import org.jetbrains.annotations.NotNull;

import java.io.*;
import java.util.List;

public class ModuleSerializer {

    public void save(@NotNull IModule module, @NotNull OutputStream outputStream) throws IOException {
        DataOutputStream dataOut = new DataOutputStream(outputStream);
        dataOut.writeUTF(module.getClass().getName());
        dataOut.writeInt(module.parameters().size());
        for (var parameter : module.parameters()) {
            saveTensor(parameter, dataOut);
        }
        dataOut.writeInt(module.subModules().size());
        for (var subModule : module.subModules()) {
            save(subModule, dataOut);
        }
    }

    private void saveTensor(@NotNull ITensor tensor, @NotNull DataOutputStream dataOut) throws IOException {
        DataType dataType = tensor.getDataType();
        dataOut.writeUTF(dataType.name());
        long[] shape = tensor.getShape();
        dataOut.writeInt(shape.length);
        for (long dim : shape) {
            dataOut.writeLong(dim);
        }
        tensor.writeTo(dataOut);
    }

    public void load(@NotNull IModule rawModule, @NotNull InputStream inputStream) throws IOException {
        DataInputStream dataIn = new DataInputStream(inputStream);
        String className = dataIn.readUTF();
        if (!className.equals(rawModule.getClass().getName())) {
            throw new IOException("Module class mismatch");
        }
        int parameterCount = dataIn.readInt();
        List<ITensor> parameters = rawModule.parameters();
        if (parameterCount != parameters.size()) {
            throw new IOException("Parameter count mismatch");
        }
        for (var parameter : parameters) {
            loadTensor(parameter, dataIn);
        }
        int subModuleCount = dataIn.readInt();
        List<IModule> subModules = rawModule.subModules();
        if (subModuleCount != subModules.size()) {
            throw new IOException("Submodule count mismatch");
        }
        for (int i = 0; i < subModuleCount; i++) {
            load(subModules.get(i), dataIn);
        }
    }

    private void loadTensor(@NotNull ITensor tensor, @NotNull DataInputStream dataIn) throws IOException {
        DataType dataType = DataType.valueOf(dataIn.readUTF());
        if (dataType != tensor.getDataType()) {
            throw new IOException("Tensor data type mismatch");
        }
        int shapeLength = dataIn.readInt();
        long[] shape = new long[shapeLength];
        for (int i = 0; i < shapeLength; i++) {
            shape[i] = dataIn.readLong();
        }
        if (!ShapeUtils.equals(shape, tensor.getShape())) {
            throw new IOException("Tensor shape mismatch");
        }
        tensor.readFrom(dataIn);
    }

}
