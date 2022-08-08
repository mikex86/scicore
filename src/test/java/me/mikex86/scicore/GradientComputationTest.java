package me.mikex86.scicore;

import me.mikex86.scicore.op.Graph;
import me.mikex86.scicore.op.IGraph;
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

    @Test
    void testTwoMalMulsBackward() {
        // Shapes:
        // (1, 4) * (4, 2) = (1, 2)
        // (1, 2) * (2, 1) = (1, 1)
        ITensor a = sciCore.matrix(new float[][]{{1, 2, 3, 4}});
        ITensor b = sciCore.matrix(new float[][]{{5, 6}, {8, 9}, {11, 13}, {15, 17}});
        ITensor c = a.matmul(b);
        assertEquals(sciCore.matrix(new float[][]{{114, 131.0f}}), c);
        ITensor d = sciCore.matrix(new float[][]{{1}, {3}});
        ITensor e = c.matmul(d);
        assertEquals(sciCore.matrix(new float[][]{{507.0f}}), e);

        IGraph graph = sciCore.getRecordedGraph();
        graph.backward();

        Graph.OperationGraphNode outputNode = (Graph.OperationGraphNode) graph.getOutputNode();
        List<IGraph.IGraphNode> inputs = outputNode.getInputs();

        assertEquals(sciCore.matrix(new float[][]{{23, 35, 50, 66}}), ((IGraph.IDifferentiableNode) inputs.get(0)).getGradient());
        assertEquals(sciCore.matrix(new float[][]{{1, 3}, {2, 6}, {3, 9}, {4, 12}}), ((IGraph.IDifferentiableNode) inputs.get(1)).getGradient());

    }

}
