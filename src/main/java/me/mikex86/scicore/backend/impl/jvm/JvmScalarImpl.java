package me.mikex86.scicore.backend.impl.jvm;

import me.mikex86.scicore.DataType;
import me.mikex86.scicore.backend.ScalarImpl;
import org.jetbrains.annotations.NotNull;

public class JvmScalarImpl implements ScalarImpl {

    @NotNull
    private final JvmScalarDataContainer dataContainer;

    JvmScalarImpl(byte value) {
        dataContainer = new JvmScalarDataContainer(value);
    }

    JvmScalarImpl(short value) {
        dataContainer = new JvmScalarDataContainer(value);
    }

    JvmScalarImpl(int value) {
        dataContainer = new JvmScalarDataContainer(value);
    }

    JvmScalarImpl(long value) {
        dataContainer = new JvmScalarDataContainer(value);
    }

    JvmScalarImpl(float value) {
        dataContainer = new JvmScalarDataContainer(value);
    }

    JvmScalarImpl(double value) {
        dataContainer = new JvmScalarDataContainer(value);
    }

    private static final class JvmScalarDataContainer {

        private byte byteValue;
        private short shortValue;
        private int intValue;
        private long longValue;
        private float floatValue;
        private double doubleValue;

        @NotNull
        private final DataType dataType;

        JvmScalarDataContainer(byte value) {
            dataType = DataType.INT8;
            byteValue = value;
        }

        JvmScalarDataContainer(short value) {
            dataType = DataType.INT16;
            shortValue = value;
        }

        JvmScalarDataContainer(int value) {
            dataType = DataType.INT32;
            intValue = value;
        }

        JvmScalarDataContainer(long value) {
            dataType = DataType.INT64;
            longValue = value;
        }

        JvmScalarDataContainer(float value) {
            dataType = DataType.FLOAT32;
            floatValue = value;
        }

        JvmScalarDataContainer(double value) {
            dataType = DataType.FLOAT64;
            doubleValue = value;
        }

        public byte getByteValue() {
            if (dataType != DataType.INT8) {
                throw new IllegalStateException("Data type is not INT8");
            }
            return byteValue;
        }

        public short getShortValue() {
            if (dataType != DataType.INT16) {
                throw new IllegalStateException("Data type is not INT16");
            }
            return shortValue;
        }

        public int getIntValue() {
            if (dataType != DataType.INT32) {
                throw new IllegalStateException("Data type is not INT32");
            }
            return intValue;
        }

        public long getLongValue() {
            if (dataType != DataType.INT64) {
                throw new IllegalStateException("Data type is not INT64");
            }
            return longValue;
        }

        public float getFloatValue() {
            if (dataType != DataType.FLOAT32) {
                throw new IllegalStateException("Data type is not FLOAT32");
            }
            return floatValue;
        }

        public double getDoubleValue() {
            if (dataType != DataType.FLOAT64) {
                throw new IllegalStateException("Data type is not FLOAT64");
            }
            return doubleValue;
        }

        @NotNull
        public DataType getDataType() {
            return dataType;
        }
    }

    @NotNull
    public DataType getDataType() {
        return this.dataContainer.getDataType();
    }

    @Override
    public byte getByte() {
        return this.dataContainer.getByteValue();
    }

    @Override
    public short getShort() {
        return this.dataContainer.getShortValue();
    }

    @Override
    public int getInt() {
        return this.dataContainer.getIntValue();
    }

    @Override
    public long getLong() {
        return this.dataContainer.getLongValue();
    }

    @Override
    public float getFloat() {
        return this.dataContainer.getFloatValue();
    }

    @Override
    public double getDouble() {
        return this.dataContainer.getDoubleValue();
    }
}
