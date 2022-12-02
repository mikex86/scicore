package me.mikex86.scicore.graph;

public enum OperationType {

    EXP, TANH, LOG, RELU, SIGMOID, // Unary arithmetic ops
    MATMUL, DIVIDE, PLUS, MINUS, POW, MULTIPLY, // Binary arithmetic ops
    REDUCE_SUM, RESHAPE, CONCAT, TRANSPOSE, // Reshape ops
    ARGMAX, COMPARE_ELEMENTS,
    ONE_HOT, GET,

    CAST,
    PLUS_INPLACE(true), MINUS_INPLACE(true); // Inplace ops (TODO: FIX)

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
