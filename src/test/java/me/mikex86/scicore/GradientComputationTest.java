package me.mikex86.scicore;

import me.mikex86.scicore.op.IGraph;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import static org.junit.jupiter.api.Assertions.*;

@TestInstance(TestInstance.Lifecycle.PER_METHOD)
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

        IGraph graph = sciCore.getGraphUpTo(result);
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
        ITensor result = a.matmul(b);

        IGraph graph = sciCore.getGraphUpTo(result);
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

        IGraph graph = sciCore.getGraphUpTo(result);
        graph.requestGradientsFor(a, b); // request for a and b
        graph.backward();

        assertFalse(graph.getGradient(result).isPresent());
    }

    /**
     * This test defines two graphs:
     * <p>
     * result1 = a * b1
     * result2 = a * b2
     * </p>
     * where the variable 'a' is shared between the two graphs.
     * The test launches two backward passes originating from result1 and result2 respectively and checks if the gradients
     * have been computed correctly.
     */
    @Test
    void testMultipleBackwardsPassesFromDifferentRootNodesWithSharedVariables() {
        ITensor a = sciCore.matrix(new float[][]{{1, 2, 3, 4, 5}});
        ITensor b1 = sciCore.matrix(new float[][]{{6}, {7}, {8}, {9}, {10}});
        ITensor b2 = sciCore.matrix(new float[][]{{11}, {12}, {13}, {14}, {15}});

        ITensor result1 = a.matmul(b1);
        ITensor result2 = a.matmul(b2);

        IGraph graph1 = sciCore.getGraphUpTo(result1);
        IGraph graph2 = sciCore.getGraphUpTo(result2);

        graph1.requestGradientsFor(a, b1);
        graph2.requestGradientsFor(a, b2);

        graph1.backward();
        graph2.backward();

        ITensor dgraph1dA = graph1.getGradient(a).orElseThrow();
        ITensor dgraph1dB1 = graph1.getGradient(b1).orElseThrow();

        ITensor dgraph2dA = graph2.getGradient(a).orElseThrow();
        ITensor dgraph2dB2 = graph2.getGradient(b2).orElseThrow();

        assertEquals(sciCore.matrix(new float[][]{{6.0f, 7.0f, 8.0f, 9.0f, 10.0f}}), dgraph1dA);
        assertEquals(sciCore.matrix(new float[][]{{1}, {2}, {3}, {4}, {5}}), dgraph1dB1);

        assertEquals(sciCore.matrix(new float[][]{{11, 12, 13, 14, 15}}), dgraph2dA);
        assertEquals(sciCore.matrix(new float[][]{{1}, {2}, {3}, {4}, {5}}), dgraph2dB2);
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
        IGraph graph = sciCore.getGraphUpTo(fwd2);

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
