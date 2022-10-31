package me.mikex86.scicore.graph.op;

import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.List;

public interface IDifferentiableSingleParametricOperation<T> extends ISingleParametricOperation<T>, IDifferentiableOperation {

    @Override
    default void computeGradients(@NotNull Graph.OperationGraphNode operationNode) {
        List<IGraph.IGraphNode> inputs = operationNode.getInputs();
        Validator.assertTrue(inputs.size() == 2, "Single-parametric operation expects one tensor and one argument (2 in total).");
        IGraph.IGraphNode input0 = inputs.get(0);
        IGraph.IGraphNode input1 = inputs.get(1);
        Validator.assertTrue(input1 instanceof IGraph.ITensorNode, "Single-parametric operation expects input1 to be hold a tensor");
        ITensor b = ((IGraph.ITensorNode) input1).getValue();
        Validator.assertTrue(b.isScalar(), "input1 of binary operation must be a scalar.");
        Class<T> tClass = getType();
        Validator.assertTrue(b.getDataType().isSameType(tClass), "input0 of binary operation must be of type " + tClass.getSimpleName() + ".");
        T t = b.element(tClass);
        ITensor upstreamGradients = operationNode.getUpstreamGradient(); // gradients with respect to z where z is the output of this operation and p1, p2...pn that are parameters of z
        Validator.assertTrue(upstreamGradients != null, "Upstream gradients not yet computed! This is a bug in DAG topology iteration!");
        computeGradients(operationNode.getOperationContext(), upstreamGradients, (IGraph.IDifferentiableNode) input0, t);
    }

    void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.IDifferentiableNode node, @Nullable T t);

}
