package me.mikex86.scicore.tests.graphviz;

import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.SciCore;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graphviz.DAGGraphRenderPlanFactory;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.condition.DisabledIf;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class GraphVisualizerTest {

    SciCore sciCore;

    @BeforeEach
    void setUp() {
        sciCore = new SciCore();
        sciCore.setBackend(SciCore.BackendType.JVM);
    }

    @Test
    @DisabledIf("me.mikex86.scicore.tests.graphviz.GraphVisualizerTest#isHeadless")
    void visualizeGraph() {
        ITensor a = sciCore.matrix(new float[][]{{1, 2, 3, 4}});
        ITensor b = sciCore.matrix(new float[][]{{5, 6}, {8, 9}, {11, 13}, {15, 17}});
        ITensor c = a.matmul(b);

        ITensor d = sciCore.matrix(new float[][]{{1}, {3}});
        ITensor e = c.matmul(d);

        IGraph graph = sciCore.getExecutionGraphUpTo(e);

        BufferedImage image = DAGGraphRenderPlanFactory.visualizeGraph(graph);
        try {
            ImageIO.write(image, "PNG", new File("graph.png"));
        } catch (IOException ex) {
            fail(ex);
        }
    }

    public static boolean isHeadless() {
        return GraphicsEnvironment.isHeadless();
    }
}