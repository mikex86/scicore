package me.mikex86.scicore.op;

import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.utils.Validator;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public interface IDifferentiableBiParametricOperation<F, S> extends IBiParametricOperation<F, S>, IDifferentiableOperation {

    @Override
    @SuppressWarnings("unchecked")
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

        Object input1Value = input1TensorNode.getValue();
        Object input2Value = input2TensorNode.getValue();

        ITensor upstreamGradients = operationNode.getGradient(); // gradients with respect to z where z is the output of this operation
        Validator.assertTrue(upstreamGradients != null, "Upstream gradients not yet computed! This is a bug in DAG topology iteration!");
        computeGradients(upstreamGradients, (IGraph.IDifferentiableNode) input0, (F) input1Value, (S) input2Value);
    }

    void computeGradients(@NotNull ITensor upstreamGradient, @NotNull IGraph.IDifferentiableNode tensor, F f, S s);

}
