package me.mikex86.scicore;

import me.mikex86.scicore.op.Graph;
import me.mikex86.scicore.op.IGraph;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class GradientComputationTest {

    SciCore sciCore;

    @BeforeEach
    void setUp() {
        sciCore = new SciCore();
        sciCore.setBackend(SciCore.BackendType.JVM);
    }

    @Test
    void testMatmulBackward() {
        ITensor a = sciCore.matrix(new float[][]{{1, 2, 3, 4, 5}});
        ITensor b = sciCore.matrix(new float[][]{{5}, {8}, {9}, {10}, {11}});
        ITensor result = a.matmul(b);
        assertEquals(sciCore.matrix(new float[][]{{143.0f}}), result);
        IGraph graph = sciCore.getRecordedGraph();
        graph.backward();
        Graph.OperationGraphNode outputNode = (Graph.OperationGraphNode) graph.getOutputNode();
        List<IGraph.IGraphNode> inputs = outputNode.getInputs();
        assertEquals(b, ((IGraph.IDifferentiableNode) inputs.get(0)).getGradient());
        assertEquals(a, ((IGraph.IDifferentiableNode) inputs.get(1)).getGradient());
    }

}
