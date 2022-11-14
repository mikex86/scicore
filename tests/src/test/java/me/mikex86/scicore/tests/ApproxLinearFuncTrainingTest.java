package me.mikex86.scicore.tests;

import kotlin.Pair;
import me.mikex86.matplotlib.jplot.JPlot;
import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.SciCore;
import me.mikex86.scicore.data.DatasetIterator;
import me.mikex86.scicore.nn.IModule;
import me.mikex86.scicore.nn.layers.Linear;
import me.mikex86.scicore.nn.optim.IOptimizer;
import me.mikex86.scicore.nn.optim.Sgd;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.tensor.ITensor;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.awt.*;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

@Disabled
@TestInstance(TestInstance.Lifecycle.PER_METHOD)
public abstract class ApproxLinearFuncTrainingTest {

    @NotNull
    private final ISciCore sciCore;

    protected ApproxLinearFuncTrainingTest(@NotNull ISciCore.BackendType backendType) {
        sciCore = new SciCore();
        sciCore.setBackend(backendType);
        sciCore.seed(123);
        sciCore.disableBackendFallback();
    }


    @NotNull
    public DatasetIterator getData(int batchSize) {
        // Function to approximate: f(x) = 2x + 0.5
        // Value range x, y in [0, 1]
        Random random = new Random(123);
        return new DatasetIterator(batchSize, () -> {
            float x = random.nextFloat();
            float y = 2 * x + 0.5f;
            ITensor featureTensor = sciCore.array(new float[]{x});
            ITensor labelTensor = sciCore.array(new float[]{y});
            return new Pair<>(featureTensor, labelTensor);
        });
    }

    @Test
    void testApproxLinearFunction() {
        float[] losses = new float[150];
        class BobNet implements IModule {

            @NotNull
            private final Linear f1 = new Linear(sciCore, DataType.FLOAT32, 1, 1, true);

            @Override
            public @NotNull ITensor forward(@NotNull ITensor input) {
                return f1.forward(input);
            }

            @Override
            public @NotNull List<IModule> subModules() {
                return List.of(f1);
            }
        }

        BobNet bobNet = new BobNet();
        int batchSize = 32;

        DatasetIterator dataIt = getData(batchSize);
        IOptimizer optimizer = new Sgd(sciCore, 0.6f, bobNet.parameters(), true, 1e-6f);
        for (int i = 0; i < 150; i++) {
            final int step = i;
            sciCore.getBackend().getOperationRecorder().recordWithScope(() -> {
                Pair<ITensor, ITensor> next = dataIt.next();
                ITensor X = next.getFirst();
                ITensor Y = next.getSecond();
                ITensor YPred = bobNet.forward(X);
                ITensor loss = (YPred.minus(Y)).pow(2).reduceSum(0).divide(batchSize);

                optimizer.step(loss);

                float lossValue = loss.elementAsFloat();
                losses[step] = (float) Math.log(lossValue);
                if (step % 10 == 0) {
                    System.out.println("Step " + step + ", loss: " + lossValue);
                }
                return null;
            });
        }
        Assertions.assertEquals(sciCore.matrix(new float[][]{{2.0f}}), bobNet.f1.getWeights());
        Assertions.assertEquals(sciCore.array(new float[]{0.5f}), bobNet.f1.getBias());

        // plot loss
        if (!GraphicsEnvironment.isHeadless()) {
            JPlot plot = new JPlot();
            plot.plot(losses, new Color(26, 188, 156), false);
            plot.setXLabel("Step");
            plot.setYLabel("Loss (log)");
            plot.save(Path.of("lin_approx_loss.png"));
        }
    }
}
