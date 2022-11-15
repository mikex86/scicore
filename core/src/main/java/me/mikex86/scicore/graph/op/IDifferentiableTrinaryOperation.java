package me.mikex86.scicore.graph.op;

import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public interface IDifferentiableTrinaryOperation extends ITrinaryOperation, IDifferentiableOperation {


    @Override
    default void computeGradients(@NotNull Graph.OperationGraphNode operationNode) {
        List<IGraph.IGraphNode> inputs = operationNode.getInputs();
        Validator.assertTrue(inputs.size() == 3, "Binary operation expects exactly two inputs");
        IGraph.IGraphNode input0 = inputs.get(0);
        IGraph.IGraphNode input1 = inputs.get(1);
        IGraph.IGraphNode input2 = inputs.get(2);

        Validator.assertTrue(input0 instanceof IGraph.ITensorNodeWithGradient, "input0 DAG node must be able to hold a gradient.");
        Validator.assertTrue(input1 instanceof IGraph.ITensorNodeWithGradient, "input1 DAG node must be able to hold a a gradient.");
        Validator.assertTrue(input2 instanceof IGraph.ITensorNodeWithGradient, "input2 DAG node must be able to hold a a gradient.");

        ITensor upstreamGradients = operationNode.getUpstreamGradient(); // gradients with respect to z where z is the output of this operation and p1, p2...pn that are parameters of z
        Validator.assertTrue(upstreamGradients != null, "Upstream gradients not yet computed! This is a bug in DAG topology iteration!");
        computeGradients(operationNode.getOperationContext(), upstreamGradients, (IGraph.ITensorNodeWithGradient) input0, (IGraph.ITensorNodeWithGradient) input1, (IGraph.ITensorNodeWithGradient) input2);
    }

    void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.ITensorNodeWithGradient a, @NotNull IGraph.ITensorNodeWithGradient b, @NotNull IGraph.ITensorNodeWithGradient c);

}
