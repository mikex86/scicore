package me.mikex86.scicore;

import me.mikex86.scicore.data.DatasetIterator;
import me.mikex86.scicore.nn.IModule;
import me.mikex86.scicore.nn.layers.Linear;
import me.mikex86.scicore.nn.layers.ReLU;
import me.mikex86.scicore.nn.optim.IOptimizer;
import me.mikex86.scicore.nn.optim.Sgd;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.utils.Pair;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.util.List;
import java.util.Random;

@TestInstance(TestInstance.Lifecycle.PER_METHOD)
public class ApproxLinearFuncTrainingTest {
    ISciCore sciCore;

    @BeforeEach
    void setUp() {
        sciCore = new SciCore();
        sciCore.setBackend(SciCore.BackendType.JVM);
        sciCore.seed(123);
    }

    @NotNull
    public DatasetIterator getBobData(int batchSize) {
        // Function to approximate: f(x) = 2x + 0.5
        // Value range x, y in [0, 1]
        Random random = new Random(123);
        return new DatasetIterator(batchSize, () -> {
            float x = random.nextFloat();
            float y = random.nextFloat();
            ITensor featureTensor = sciCore.array(new float[]{x});
            ITensor labelTensor = sciCore.array(new float[]{y});
            return Pair.of(featureTensor, labelTensor);
        });
    }

    @Test
    void testEasyExample() {

        class BobNet implements IModule {

            @NotNull
            private final Linear f1 = new Linear(sciCore, DataType.FLOAT32, 1, 1, true);

            @Override
            public @NotNull ITensor forward(@NotNull ITensor input) {
                return f1.forward(input);
            }

            @Override
            public @NotNull List<ITensor> parameters() {
                return collectParameters(f1);
            }
        }

        BobNet bobNet = new BobNet();
        int batchSize = 32;

        DatasetIterator dataIt = getBobData(batchSize);
        IOptimizer optimizer = new Sgd(sciCore, 0.01f, bobNet.parameters());
        for (int step = 0; step < 10000; step++) {
            Pair<ITensor, ITensor> next = dataIt.next();
            ITensor X = next.getFirst();
            ITensor Y = next.getSecond();
            ITensor YPred = bobNet.forward(X);
            ITensor loss = (YPred.minus(Y)).pow(2).reduceSum(0).divided(batchSize);

            IGraph graph = sciCore.getGraphUpTo(loss);
            optimizer.step(graph);

            if (step % 100 == 0) {
                System.out.println("Step " + step + ", loss: " + loss.elementAsFloat());
            }
        }
    }

}
