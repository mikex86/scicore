package me.mikex86.scicore.graph;

import org.jetbrains.annotations.NotNull;

import static me.mikex86.scicore.graph.OperationType.Arity.*;
import static me.mikex86.scicore.graph.OperationType.Category.*;

public enum OperationType {

    EXP(ARITHMETIC, UNARY), TANH(ARITHMETIC, UNARY), LOG(ARITHMETIC, UNARY), RELU(ARITHMETIC, UNARY), SIGMOID(ARITHMETIC, UNARY), // Unary arithmetic ops
    MATMUL(ARITHMETIC, BINARY), DIVIDE(ARITHMETIC, BINARY), PLUS(ARITHMETIC, BINARY), MINUS(ARITHMETIC, BINARY), POW(ARITHMETIC, BINARY), MULTIPLY(ARITHMETIC, BINARY), // Binary arithmetic ops
    REDUCE_SUM(Category.RESHAPE, GENERIC), RESHAPE(Category.RESHAPE, GENERIC), TRANSPOSE(Category.RESHAPE, GENERIC), // Reshape ops
    ARGMAX(INDEXING, GENERIC), ONE_HOT(INDEXING, GENERIC), GET(INDEXING, GENERIC), WHERE(INDEXING, GENERIC), // Indexing ops
    CONCAT(COPY, GENERIC), STACK(COPY, GENERIC), // Copy ops
    COMPARE_ELEMENTS(COMPARISON, BINARY), LESS_THAN(COMPARISON, BINARY), // Comparison ops
    CAST(MISC, UNARY), // Misc ops

    FILL_TRIANGLE(FILL, GENERIC), // Fill ops

    PLUS_INPLACE(ARITHMETIC, BINARY, true), MINUS_INPLACE(ARITHMETIC, BINARY, true); // Inplace ops (TODO: FIX)


    public enum Category {
        ARITHMETIC, RESHAPE, INDEXING, COPY, COMPARISON, MISC, FILL, INPLACE
    }

    public enum Arity {
        UNARY, BINARY, GENERIC
    }


    @NotNull
    private final OperationType.Category category;

    @NotNull
    private final Arity arity;

    private final boolean inplace;

    OperationType(@NotNull OperationType.Category category, @NotNull Arity arity, boolean inplace) {
        this.category = category;
        this.arity = arity;
        this.inplace = inplace;
    }

    OperationType(Category category, @NotNull Arity arity) {
        this(category, arity, false);
    }


    @NotNull
    public OperationType.Category getCategory() {
        return category;
    }

    @NotNull
    public Arity getArity() {
        return arity;
    }

    public boolean isInplace() {
        return inplace;
    }

}
