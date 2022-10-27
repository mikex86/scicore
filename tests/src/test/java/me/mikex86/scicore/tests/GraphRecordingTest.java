package me.mikex86.scicore.tests;

import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.SciCore;
import me.mikex86.scicore.graph.Graph;
import me.mikex86.scicore.graph.IGraph;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;

@Disabled // this test is disabled because it is abstract
abstract class GraphRecordingTest {

    ISciCore sciCore;

    public GraphRecordingTest(@NotNull ISciCore.BackendType backendType) {
        sciCore = new SciCore();
        sciCore.setBackend(backendType);
    }

    /**
     * This tests records operations which compute two completely independent outputs, each of which can be the root node of a separate graph.
     */
    @Test
    void testRecordingOfIndependentGraphs() {
        ITensor a1 = sciCore.matrix(new float[][]{{1, 2, 3, 4, 5}});
        ITensor b1 = sciCore.matrix(new float[][]{{6}, {7}, {8}, {9}, {10}});
        ITensor result1 = a1.matmul(b1);

        ITensor a2 = sciCore.matrix(new float[][]{{10, 11, 12, 13, 14}});
        ITensor b2 = sciCore.matrix(new float[][]{{15}, {16}, {17}, {18}, {19}});
        ITensor result2 = a2.matmul(b2);

        IGraph graph1 = sciCore.getGraphUpTo(result1);
        assertEquals(result1, ((IGraph.ITensorNode) graph1.getOutputNode()).getValue());
        assertEquals(List.of(a1, b1), ((Graph.OperationGraphNode) graph1.getOutputNode()).getInputs().stream().map(n -> ((IGraph.ITensorNode) n).getValue()).collect(Collectors.toList()));

        IGraph graph2 = sciCore.getGraphUpTo(result2);
        assertEquals(result2, ((IGraph.ITensorNode) graph2.getOutputNode()).getValue());
        assertEquals(List.of(a2, b2), ((Graph.OperationGraphNode) graph2.getOutputNode()).getInputs().stream().map(n -> ((IGraph.ITensorNode) n).getValue()).collect(Collectors.toList()));
    }


}
