package me.mikex86.scicore.op;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public class ParameterBundle {

    @NotNull
    private final List<ITensor> parameters;

    public ParameterBundle(@NotNull List<ITensor> parameters) {
        this.parameters = parameters;
    }

    @NotNull
    public ITensor getParameter(int parameterIndex) {
        if (parameterIndex < 0 || parameterIndex > parameters.size()) {
            throw new IllegalArgumentException("Parameter index out of bounds: " + parameterIndex + " (size: " + parameters.size() + ")");
        }
        return parameters.get(parameterIndex);
    }

    public byte getParameterInt8(int parameterIndex) {
        ITensor parameter = getParameter(parameterIndex);
        if (parameter.getDataType() != DataType.INT8) {
            throw new IllegalArgumentException("Parameter " + parameterIndex + " is not of type UINT8");
        }
        return parameter.elementAsByte();
    }

    public short getParameterInt16(int parameterIndex) {
        ITensor parameter = getParameter(parameterIndex);
        if (parameter.getDataType() != DataType.INT16) {
            throw new IllegalArgumentException("Parameter " + parameterIndex + " is not of type UINT16");
        }
        return parameter.elementAsShort();
    }

    public int getParameterInt32(int parameterIndex) {
        ITensor parameter = getParameter(parameterIndex);
        if (parameter.getDataType() != DataType.INT32) {
            throw new IllegalArgumentException("Parameter " + parameterIndex + " is not of type UINT32");
        }
        return parameter.elementAsInt();
    }

    public long getParameterInt64(int parameterIndex) {
        ITensor parameter = getParameter(parameterIndex);
        if (parameter.getDataType() != DataType.INT64) {
            throw new IllegalArgumentException("Parameter " + parameterIndex + " is not of type UINT64");
        }
        return parameter.elementAsLong();
    }

    public float getParameterFloat32(int parameterIndex) {
        ITensor parameter = getParameter(parameterIndex);
        if (parameter.getDataType() != DataType.FLOAT32) {
            throw new IllegalArgumentException("Parameter " + parameterIndex + " is not of type FLOAT32");
        }
        return parameter.elementAsFloat();
    }

    public double getParameterFloat64(int parameterIndex) {
        ITensor parameter = getParameter(parameterIndex);
        if (parameter.getDataType() != DataType.FLOAT64) {
            throw new IllegalArgumentException("Parameter " + parameterIndex + " is not of type FLOAT64");
        }
        return parameter.elementAsDouble();
    }

    public boolean getParameterBool(int parameterIndex) {
        ITensor parameter = getParameter(parameterIndex);
        if (parameter.getDataType() != DataType.BOOLEAN) {
            throw new IllegalArgumentException("Parameter " + parameterIndex + " is not of type BOOL");
        }
        return parameter.elementAsBoolean();
    }

    @NotNull
    public <T extends Enum<T>> T getParameterEnum(int parameterIndex, @NotNull Class<T> enumClass) {
        ITensor parameter = getParameter(parameterIndex);
        if (parameter.getDataType() != DataType.INT32) {
            throw new IllegalArgumentException("Enum parameters must be of type INT32");
        }
        T[] enumConstants = enumClass.getEnumConstants();
        if (enumConstants == null) {
            throw new IllegalArgumentException("Enum class " + enumClass.getName() + " has no constants");
        }
        int enumIndex = parameter.elementAsInt();
        if (enumIndex < 0 || enumIndex >= enumConstants.length) {
            throw new IllegalArgumentException("Enum index out of bounds: " + enumIndex + " (size: " + enumConstants.length + ")");
        }
        return enumConstants[enumIndex];
    }
}
