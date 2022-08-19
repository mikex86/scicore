package me.mikex86.scicore.op;

import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public interface IDifferentiableBinaryOperation extends IBinaryOperation, IDifferentiableOperation {

    @Override
    default void computeGradients(@NotNull Graph.OperationGraphNode operationNode) {
        List<IGraph.IGraphNode> inputs = operationNode.getInputs();
        Validator.assertTrue(inputs.size() == 2, "Binary operation expects exactly two inputs");
        IGraph.IGraphNode input0 = inputs.get(0);
        IGraph.IGraphNode input1 = inputs.get(1);

        Validator.assertTrue(input0 instanceof IGraph.IDifferentiableNode, "Binary operation expects input0 to be a IDifferentiableNode.");
        Validator.assertTrue(input1 instanceof IGraph.IDifferentiableNode, "Binary operation expects input1 to be a IDifferentiableNode.");

        ITensor upstreamGradient = operationNode.getGradient(); // gradient with respect to z where z is the output of the operation
        Validator.assertTrue(upstreamGradient != null, "Upstream gradient not yet computed! This is a bug in DAG topology iteration!");
        computeGradients(upstreamGradient, (IGraph.IDifferentiableNode) input0, (IGraph.IDifferentiableNode) input1);
    }

    void computeGradients(@NotNull ITensor upstreamGradient, @NotNull IGraph.IDifferentiableNode a, @NotNull IGraph.IDifferentiableNode b);

}
