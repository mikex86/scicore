package me.mikex86.scicore.op;

import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public interface IDifferentiableUnaryOperation extends IUnaryOperation, IDifferentiableOperation {

    @Override
    default void computeGradients(@NotNull Graph.OperationGraphNode operationNode) {
        List<IGraph.IGraphNode> inputs = operationNode.getInputs();
        Validator.assertTrue(inputs.size() == 1, "Unary operation expects exactly one input.");
        IGraph.IGraphNode input0 = inputs.get(0);

        Validator.assertTrue(input0 instanceof IGraph.ITensorNodeWithGradient, "input0 DAG node must be able to hold a gradient.");

        ITensor upstreamGradient = operationNode.getGradient(); // gradient with respect to z where z is the output of the operation
        Validator.assertTrue(upstreamGradient != null, "Upstream gradient not yet computed! This is a bug in DAG topology iteration!");
        computeGradients(upstreamGradient, (IGraph.IDifferentiableNode) input0);
    }

    void computeGradients(@NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient input);

}
