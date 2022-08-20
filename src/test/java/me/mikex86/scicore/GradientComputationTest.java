package me.mikex86.scicore;

import me.mikex86.scicore.op.IGraph;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

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
        ITensor b = sciCore.matrix(new float[][]{{6}, {7}, {8}, {9}, {10}});
        ITensor result = a.matmul(b);
        assertEquals(sciCore.matrix(new float[][]{{130.0f}}), result);

        IGraph graph = sciCore.getRecordedGraph();
        graph.requestGradientsFor(a, b);
        graph.backward();

        ITensor dLdA = graph.getGradient(a).orElseThrow();
        ITensor dLdB = graph.getGradient(b).orElseThrow();

        assertEquals(sciCore.matrix(new float[][]{{6.0f, 7.0f, 8.0f, 9.0f, 10.0f}}), dLdA);
        assertEquals(sciCore.matrix(new float[][]{{1}, {2}, {3}, {4}, {5}}), dLdB);
    }

    @Test
    void testOnlyComputeGradientForRequested() {
        ITensor a = sciCore.matrix(new float[][]{{1, 2, 3, 4, 5}});
        ITensor b = sciCore.matrix(new float[][]{{6}, {7}, {8}, {9}, {10}});
        a.matmul(b);

        IGraph graph = sciCore.getRecordedGraph();
        graph.requestGradientsFor(a); // only request for a

        graph.backward();

        assertTrue(graph.getGradient(a).isPresent());
        assertFalse(graph.getGradient(b).isPresent());
    }

    @Test
    void testGradientsOnlySavedForExplicitlyRequested() {
        ITensor a = sciCore.matrix(new float[][]{{1, 2, 3, 4, 5}});
        ITensor b = sciCore.matrix(new float[][]{{6}, {7}, {8}, {9}, {10}});
        ITensor result = a.matmul(b);

        IGraph graph = sciCore.getRecordedGraph();
        graph.requestGradientsFor(a, b); // request for a and b
        graph.backward();

        assertFalse(graph.getGradient(result).isPresent());
    }

    @Test
    void testMatMulChainRuleBackward() {
        // See: https://cs231n.github.io/optimization-2/#mat (Stanford University CS231n: Deep Learning for Computer Vision)

        /*
          # Neural Network

          input = (2, n) # (n_inputs: 2, batch_size: n)

          # Layer 1
          w1 = (4, 2) # neurons: 4, input_size: 2
          fwd1 = w1 * input
          fwd1 = (4, 2) * (2, n)
          fwd1 = (4, n)

          # Layer 2
          w2 = (1, 4) # neurons: 1, input_size: 4
          fwd2 = w2 * fwd1
          fwd2 = (1, 4) * (4, n)
          fwd2 = (1, n) # (output: 1, batch_size: n)

         */
        ITensor act = sciCore.matrix(new float[][]{{1}, {2}}); // (2, n)
        ITensor w1 = sciCore.matrix(new float[][]{{3, 4}, {5, 6}, {7, 8}, {9, 10}});
        ITensor w2 = sciCore.matrix(new float[][]{{11, 12, 13, 14}});

        ITensor fwd1 = w1.matmul(act);
        ITensor fwd2 = w2.matmul(fwd1);

        assertEquals(sciCore.matrix(new float[][]{{1030.f}}), fwd2); // check forward pass

        // Automatic backpropagation
        IGraph graph = sciCore.getRecordedGraph();

        graph.requestGradientsFor(w1, w2);

        graph.backward();

        // Manual backpropagation
        // dL/dFwd2 = 1
        ITensor dL_dFwd2 = sciCore.onesLike(fwd2);

        // dL/dW2 = dL/dFwd2 * dFwd2/dW2
        // dFwd2/dW2 = fwd1.T
        ITensor dFwd2_dW2 = fwd1.transpose(); // local gradient
        ITensor dL_dW2 = dL_dFwd2.matmul(dFwd2_dW2);

        // dL/dFwd1 = w2.T * dL/dFwd2
        ITensor dL_dFwd1 = w2.transpose().matmul(dL_dFwd2);

        // dL/dW1 = dL/dFwd1 * dFwd1/dW1
        // dFwd1/dW1 = act.T
        ITensor dFwd1_dW1 = act.transpose(); // local gradient
        ITensor dL_dW1 = dL_dFwd1.matmul(dFwd1_dW1);

        // Check automatic backpropagation against manual backpropagation
        ITensor dL_dW2_automatic = graph.getGradient(w2).orElseThrow();
        ITensor dL_dW1_automatic = graph.getGradient(w1).orElseThrow();

        assertEquals(dL_dW2, dL_dW2_automatic);
        assertEquals(dL_dW1, dL_dW1_automatic);
    }

}
