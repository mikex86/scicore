package me.mikex86.scicore;

import me.mikex86.matplotlib.jplot.JPlot;
import me.mikex86.scicore.data.DatasetIterator;
import me.mikex86.scicore.nn.IModule;
import me.mikex86.scicore.nn.layers.Linear;
import me.mikex86.scicore.nn.layers.Sigmoid;
import me.mikex86.scicore.nn.optim.IOptimizer;
import me.mikex86.scicore.nn.optim.Sgd;
import me.mikex86.scicore.op.IGraph;
import me.mikex86.scicore.utils.Pair;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.awt.*;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

@TestInstance(TestInstance.Lifecycle.PER_METHOD)
public class ApproxNonlinearFuncTrainingTest {

    ISciCore sciCore;

    @BeforeEach
    void setUp() {
        sciCore = new SciCore();
        sciCore.setBackend(SciCore.BackendType.JVM);
        sciCore.seed(123);
    }

    @NotNull
    public DatasetIterator getData(int batchSize) {
        // Function to approximate: f(x) = 2x^2 + 0.5
        // Value range x, y in [0, 1]
        Random random = new Random(123);
        return new DatasetIterator(batchSize, () -> {
            float x = random.nextFloat();
            float y = 2 * (x * x) + 0.5f;
            ITensor featureTensor = sciCore.array(new float[]{x});
            ITensor labelTensor = sciCore.array(new float[]{y});
            return Pair.of(featureTensor, labelTensor);
        });
    }

    @Test
    void testNonLinearFunc() {

        class BobNet implements IModule {

            @NotNull
            private final Sigmoid act = new Sigmoid();

            @NotNull
            private final Linear f1 = new Linear(sciCore, DataType.FLOAT32, 1, 1, true);

            @NotNull
            private final Linear f2 = new Linear(sciCore, DataType.FLOAT32, 1, 1, true);

            @Override
            public @NotNull ITensor forward(@NotNull ITensor input) {
                ITensor out = f1.forward(input);
                out = act.forward(out);
                out = f2.forward(out);
                return out;
            }

            @Override
            public @NotNull List<ITensor> parameters() {
                return collectParameters(f1, f2);
            }
        }

        float[] losses = new float[250];

        BobNet bobNet = new BobNet();
        int batchSize = 64;

        DatasetIterator dataIt = getData(batchSize);
        IOptimizer optimizer = new Sgd(sciCore, 0.4f, bobNet.parameters(), true, 0.99999f);
        for (int step = 0; step < 250; step++) {
            sciCore.getBackend().getOperationRecorder().resetRecording();
            Pair<ITensor, ITensor> next = dataIt.next();
            ITensor X = next.getFirst();
            ITensor Y = next.getSecond();
            ITensor YPred = bobNet.forward(X);
            ITensor loss = (YPred.minus(Y)).pow(2).reduceSum(0).divided(batchSize);

            IGraph graph = sciCore.getGraphUpTo(loss);
            optimizer.step(graph);

            float lossValue = loss.elementAsFloat();
            losses[step] = (float) Math.log(lossValue);
            if (step % 10 == 0) {
                System.out.println("Step " + step + ", loss: " + lossValue);
            }
        }

        // plot loss
        {
            JPlot plot = new JPlot();
            plot.plot(losses, new Color(26, 188, 156), false);
            plot.setXLabel("Step");
            plot.setYLabel("Loss (log)");
            plot.save(Path.of("non_lin_approx_loss.png"));
        }
    }

}
