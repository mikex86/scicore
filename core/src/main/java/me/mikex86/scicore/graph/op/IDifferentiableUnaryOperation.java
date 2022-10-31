package me.mikex86.scicore.graph.op;

import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
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

        ITensor upstreamGradients = operationNode.getUpstreamGradient(); // gradients with respect to z where z is the output of this operation and p1, p2...pn that are parameters of z
        Validator.assertTrue(upstreamGradients != null, "Upstream gradient not yet computed! This is a bug in DAG topology iteration!");
        computeGradients(operationNode.getOperationContext(), upstreamGradients, (IGraph.ITensorNodeWithGradient) input0);
    }

    void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient input);

}
