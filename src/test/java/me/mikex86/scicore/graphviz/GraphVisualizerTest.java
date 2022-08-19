package me.mikex86.scicore.graphviz;

import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.SciCore;
import me.mikex86.scicore.op.IGraph;
import org.jetbrains.skija.EncodedImageFormat;
import org.jetbrains.skija.Image;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.ByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Objects;

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

        IGraph graph = sciCore.getRecordedGraph();

        Image image = GraphVisualizer.visualizeGraph(graph);

        try {
            ByteChannel channel = Files.newByteChannel(
                    Path.of("output.png"),
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE);
            channel.write(ByteBuffer.wrap(Objects.requireNonNull(image.encodeToData(EncodedImageFormat.PNG)).getBytes()));
            channel.close();
        } catch (IOException ex) {
            fail(ex);
        }
    }
}