package me.mikex86.scicore.tests;

import kotlin.Pair;
import me.mikex86.matplotlib.jplot.JPlot;
import me.mikex86.scicore.SciCore;
import me.mikex86.scicore.tensor.DataType;
import me.mikex86.scicore.ISciCore;
import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.data.DatasetIterator;
import me.mikex86.scicore.nn.IModule;
import me.mikex86.scicore.nn.layers.Linear;
import me.mikex86.scicore.nn.act.Sigmoid;
import me.mikex86.scicore.nn.optim.IOptimizer;
import me.mikex86.scicore.nn.optim.Sgd;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.awt.*;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

@Disabled
@TestInstance(TestInstance.Lifecycle.PER_METHOD)
public abstract class ApproxNonlinearFuncTrainingTest {

    @NotNull
    private final ISciCore sciCore;

    protected ApproxNonlinearFuncTrainingTest(@NotNull ISciCore.BackendType backendType) {
        sciCore = new SciCore();
        sciCore.setBackend(backendType);
        sciCore.seed(123);
        sciCore.disableBackendFallback();
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
            return new Pair<>(featureTensor, labelTensor);
        });
    }

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
        public @NotNull List<IModule> subModules() {
            return List.of(f1, f2);
        }
    }

    @Test
    void testNonLinearFunc() {
        int nSteps = 2_000;
        float[] losses = new float[nSteps];
        BobNet bobNet = new BobNet();
        int batchSize = 32;

        DatasetIterator dataIt = getData(batchSize);
        IOptimizer optimizer = new Sgd(sciCore, 0.5f, 0.05f, nSteps, bobNet.parameters());
        for (int i = 0; i < nSteps; i++) {
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
                if (step % 100 == 0) {
                    System.out.println("Step " + step + ", loss: " + lossValue);
                    if (!GraphicsEnvironment.isHeadless()) {
                        plotPrediction(bobNet);
                    }
                }
                return null;
            });
        }

        // plot loss
        if (!GraphicsEnvironment.isHeadless()) {
            JPlot plot = new JPlot();
            plot.plot(losses, new Color(26, 188, 156), false);
            plot.setXLabel("Step");
            plot.setYLabel("Loss (log)");
            plot.save(Path.of("non_lin_approx_loss.png"));
        }
    }

    private JPlot plot;

    private void plotPrediction(@NotNull BobNet bobNet) {
        if (plot == null) {
            plot = new JPlot();
        }
        int nPoints = 100;
        float[] modelPredictionY = new float[nPoints];
        float[] realY = new float[nPoints];
        float intervalStart = 0, intervalEnd = 1;
        ITensor X = sciCore.arange(intervalStart, intervalEnd, (intervalEnd - intervalStart) / (float) (nPoints - 1), DataType.FLOAT32).getReshapedView(new long[]{nPoints, 1});
        ITensor YPred = bobNet.forward(X);
        for (int i = 0; i < nPoints; i++) {
            modelPredictionY[i] = YPred.getFloat(i, 0);
            float x = i / (float) nPoints;
            float y = 2 * (x * x) + 0.5f;
            realY[i] = y;
        }
        plot.clear();
        plot.setBackgroundColor(new Color(36, 36, 36));
        plot.plot(modelPredictionY, new Color(26, 188, 156), true);
        plot.plot(realY, new Color(46, 204, 113), false);
        plot.setXLabel("X");
        plot.setYLabel("Y");
        plot.show(false);
    }

}
