package me.mikex86.scicore.op;

import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.List;

public interface IDifferentiableBiParametricOperation<F, S> extends IBiParametricOperation<F, S>, IDifferentiableOperation {

    @Override
    default void computeGradients(@NotNull Graph.OperationGraphNode operationNode) {
        List<IGraph.IGraphNode> inputs = operationNode.getInputs();
        Validator.assertTrue(inputs.size() == 3, "Bi-parametric operation expects one tensor and two arguments (3 in total).");
        IGraph.IGraphNode input0 = inputs.get(0);
        IGraph.IGraphNode input1 = inputs.get(1);
        IGraph.IGraphNode input2 = inputs.get(2);

        if (!(input1 instanceof IGraph.ITensorNode input1TensorNode)) {
            throw new IllegalArgumentException("Bi-parametric operation expects input1 to be hold a tensor");
        }
        if (!(input2 instanceof IGraph.ITensorNode input2TensorNode)) {
            throw new IllegalArgumentException("Bi-parametric operation expects input2 to hold a tensor");
        }

        ITensor b = input1TensorNode.getValue();
        ITensor c = input2TensorNode.getValue();

        Validator.assertTrue(b.isScalar(), "input1 of binary operation must be a scalar.");
        Validator.assertTrue(c.isScalar(), "input2 of binary operation must be a scalar.");

        Class<F> fClass = getFirstType();
        Class<S> sClass = getSecondType();

        Validator.assertTrue(b.getDataType().isSameType(fClass), "input0 of binary operation must be of type " + fClass.getSimpleName() + ".");
        Validator.assertTrue(c.getDataType().isSameType(sClass), "input1 of binary operation must be of type " + sClass.getSimpleName() + ".");

        F f = b.element(fClass);
        S s = c.element(sClass);

        ITensor upstreamGradients = operationNode.getUpstreamGradient(); // gradients with respect to z where z is the output of this operation and p1, p2...pn that are parameters of z
        Validator.assertTrue(upstreamGradients != null, "Upstream gradients not yet computed! This is a bug in DAG topology iteration!");
        computeGradients(operationNode.getOperationContext(), upstreamGradients, (IGraph.IDifferentiableNode) input0, f, s);
    }

    void computeGradients(@NotNull Graph.IOperationContext ctx, @NotNull ITensor upstreamGradient, @NotNull IGraph.IDifferentiableNode node, @Nullable F f, @Nullable S s);

}
