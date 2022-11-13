package me.mikex86.scicore.graph;

public enum OperationType {

    MATMUL, DIVIDE, PLUS, MINUS, REDUCE_SUM, EXP, TRANSPOSE, POW, MULTIPLY, RELU, SIGMOID, ARGMAX, CAST, COMPARE_ELEMENTS,
    PLUS_INPLACE(true), MINUS_INPLACE(true);

    private final boolean inplace;

    OperationType(boolean inplace) {
        this.inplace = inplace;
    }

    OperationType() {
        this(false);
    }

    public boolean isInplace() {
        return inplace;
    }

}
