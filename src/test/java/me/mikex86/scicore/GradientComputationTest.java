package me.mikex86.scicore;

import me.mikex86.scicore.op.IGraph;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import static org.junit.jupiter.api.Assertions.*;

@TestInstance(TestInstance.Lifecycle.PER_METHOD)
public class GradientComputationTest {

    ISciCore sciCore;

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

    @Test
    void testMatmulAndPlusBackward() {
        ITensor a = sciCore.matrix(new float[][]{{1, 2}});
        ITensor b = sciCore.matrix(new float[][]{{3, 4}, {5, 6}});
        ITensor c = sciCore.matrix(new float[][]{{7, 8}});
        ITensor d = sciCore.matrix(new float[][]{{9}, {10}});

        ITensor e = a.matmul(b);
        ITensor f = e.plus(c);
        ITensor g = f.matmul(d);

        IGraph graph = sciCore.getGraphUpTo(g);
        graph.requestGradientsFor(a, b, c, d);
        graph.backward();

        ITensor dLdA = graph.getGradient(a).orElseThrow();
        ITensor dLdB = graph.getGradient(b).orElseThrow();
        ITensor dLdC = graph.getGradient(c).orElseThrow();
        ITensor dLdD = graph.getGradient(d).orElseThrow();

        assertEquals(sciCore.matrix(new float[][]{{67.0f, 105.0f}}), dLdA);
        assertEquals(sciCore.matrix(new float[][]{{9.0f, 10.0f}, {18.0f, 20.0f}}), dLdB);
        assertEquals(sciCore.matrix(new float[][]{{9.0f, 10.0f}}), dLdC);
        assertEquals(sciCore.matrix(new float[][]{{20.0f}, {24.0f}}), dLdD);
    }

    @Test
    void testElementWiseMultiply() {
        ITensor a = sciCore.matrix(new float[][]{{2, 3}});
        ITensor b = sciCore.matrix(new float[][]{{10, 11}});
        ITensor c = sciCore.matrix(new float[][]{{2}, {3}});
        ITensor d = a.multiply(b); // element-wise multiplication
        ITensor e = d.matmul(c); // matrix multiplication to get a scalar

        IGraph graph = sciCore.getGraphUpTo(e);
        graph.requestGradientsFor(a, b);
        graph.backward();

        ITensor dLdA = graph.getGradient(a).orElseThrow();
        ITensor dLdB = graph.getGradient(b).orElseThrow();

        assertEquals(sciCore.matrix(new float[][]{{20.0f, 33.0f}}), dLdA);
        assertEquals(sciCore.matrix(new float[][]{{4.0f, 9.0f}}), dLdB);
    }

    @Test
    void testPowWithOneElement() {
        ITensor a = sciCore.matrix(new float[][]{{5}});
        ITensor b = sciCore.scalar(2);

        ITensor c = a.pow(b);

        assertEquals(sciCore.matrix(new float[][]{{25.0f}}), c);

        IGraph graph = sciCore.getGraphUpTo(c);
        graph.requestGradientsFor(a, b);
        graph.backward();

        ITensor dLdA = graph.getGradient(a).orElseThrow();
        ITensor dLdB = graph.getGradient(b).orElseThrow();

        assertEquals(sciCore.matrix(new float[][]{{10.0f}}), dLdA);
        assertEquals(40.2359f, dLdB.elementAsFloat(), 0.0001f);
    }

    @Test
    void testPowWithMultipleElements() {
        // (1, 8) * (8, 1) = (1, 1)
        ITensor a = sciCore.matrix(new float[][]{{2, 3, 4, 5, 6, 7, 8, 9}});
        ITensor b = sciCore.scalar(3);
        ITensor c = sciCore.matrix(new float[][]{{4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}});
        ITensor d = a.pow(b);
        ITensor e = d.matmul(c);

        IGraph graph = sciCore.getGraphUpTo(e);
        graph.requestGradientsFor(a, b);
        graph.backward();

        ITensor dLdA = graph.getGradient(a).orElseThrow();
        ITensor dLdB = graph.getGradient(b).orElseThrow();

        assertEquals(sciCore.matrix(new float[][]{{48.0f, 135.0f, 288.0f, 525.0f, 864.0f, 1323.0f, 1920.0f, 2673.0f}}), dLdA);
        assertEquals(39480.5586f, dLdB.elementAsFloat(), 0.0001f);
    }

    @Test
    void testMatmulWithReduceSum() {
        // (1, 2) * (2, 2) = (1, 2)
        ITensor a = sciCore.matrix(new float[][]{{1, 2}});
        ITensor b = sciCore.matrix(new float[][]{{3, 4}, {5, 6}});
        ITensor c = a.matmul(b);
        ITensor d = c.reduceSum(1);

        IGraph graph = sciCore.getGraphUpTo(d);
        graph.requestGradientsFor(a, b);
        graph.backward();

        ITensor dLdA = graph.getGradient(a).orElseThrow();
        ITensor dLdB = graph.getGradient(b).orElseThrow();

        assertEquals(sciCore.matrix(new float[][]{{7.0f, 11.0f}}), dLdA);
        assertEquals(sciCore.matrix(new float[][]{{1.0f, 1.0f}, {2.0f, 2.0f}}), dLdB);
    }

    /**
     * This test aims to test the totality of a default linear neural network without activation
     * function as it is composed inside a loss function and reduction to a single loss scalar that is differentiated
     * in respect to.
     */
    @Test
    void testMatmulAndPlusWithSquareReduceSum() {
        // batch_size = n = 3
        // X = input = (4, n)
        // W = weights = (2, 4)
        // D = WX = (2, 4) * (4, n) = (2, n)
        // Y = expected output = (2, n)
        // L = ((Y - D)^2).sum(dim=0)

        ITensor X = sciCore.matrix(new float[][]{{0.1f, 0.3f, 0.9f, 0.3f}, {0.4f, 0.01f, 0.23f, 0.93f}, {0.93f, 0.5f, 0.9f, 0.44f}}).transpose();
        ITensor W = sciCore.matrix(new float[][]{{0.4f, 0.2f, 0.34f, 0.24f}, {0.5f, 0.36f, 0.67f, 0.38f}});
        ITensor Y = sciCore.matrix(new float[][]{{0.93f, 0.42f, 0.94f}, {0.1f, 0.5f, 0.24f}});

        ITensor D = W.matmul(X);
        ITensor diff = Y.minus(D);
        ITensor squared = sciCore.pow(diff, 2);
        ITensor lossPerSample = squared.reduceSum(0);
        ITensor loss = lossPerSample.reduceSum(0);

        IGraph graph = sciCore.getGraphUpTo(loss);
        graph.requestGradientsFor(W);

        graph.backward();

        ITensor dLdW = graph.getGradient(W).orElseThrow();
        assertEquals(sciCore.matrix(new float[][]{
                {-0.1606f, -0.3267f, -0.8952f, -0.2401f},
                {2.5098f, 1.6444f, 3.6075f, 1.8918f}
        }), dLdW);
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

}
