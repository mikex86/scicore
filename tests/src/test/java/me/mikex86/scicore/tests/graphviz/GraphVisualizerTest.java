package me.mikex86.scicore.tests.graphviz;

import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.SciCore;
import me.mikex86.scicore.graph.IGraph;
import me.mikex86.scicore.graphviz.DAGGraphRenderPlanFactory;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import javax.imageio.ImageIO;
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
    void visualizeGraph() {
        ITensor a = sciCore.matrix(new float[][]{{1, 2, 3, 4}});
        ITensor b = sciCore.matrix(new float[][]{{5, 6}, {8, 9}, {11, 13}, {15, 17}});
        ITensor c = a.matmul(b);

        assertEquals(sciCore.matrix(new float[][]{{114, 131.0f}}), c);

        ITensor d = sciCore.matrix(new float[][]{{1}, {3}});
        ITensor e = c.matmul(d);

        assertEquals(sciCore.matrix(new float[][]{{507.0f}}), e);

        IGraph graph = sciCore.getGraphUpTo(e);

        BufferedImage image = DAGGraphRenderPlanFactory.visualizeGraph(graph);

        try {
            ImageIO.write(image, "PNG", new File("graph.png"));
        } catch (IOException ex) {
            fail(ex);
        }
    }
}