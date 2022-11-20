<p align="center">
    <img src="logo_with_text.png" alt="SciCore Logo" width="35%" />
</p>

[![Java CI with Gradle](https://github.com/mikex86/scicore/actions/workflows/gradle.yml/badge.svg)](https://github.com/mikex86/scicore/actions/workflows/gradle.yml)

SciCore is a tiny tensor processing library and autograd engine with a focus on simplicity. It is written in Java,
with some native code for hardware specific optimizations.
At the moment the only fully featured backend is the jvm backend, which is a pure java implementation.
SciCore is very experimental and not ready for production use and primarily serves as a playground to explore the
fundamentals
of deep learning with a tensor processing library that has shallow levels of abstraction.

## Simple MNIST Example
    
```kotlin
class MnistNet(sciCore: ISciCore) : IModule {

    private val act = ReLU()
    private val fc1 = Linear(sciCore, DataType.FLOAT32, (28 * 28).toLong(), 128, true)
    private val fc2 = Linear(sciCore, DataType.FLOAT32, 128, 10, true)
    private val softmax = Softmax(sciCore, 1)

    override fun forward(input: ITensor): ITensor {
        return fc1(input)
            .use { h -> act(h) }
            .use { h -> fc2(h) }
            .use { h -> softmax(h) }
    }

    override fun subModules(): List<IModule> {
        return listOf(fc1, fc2)
    }
}

fun main() {
    val sciCore = SciCore()
    sciCore.setBackend(ISciCore.BackendType.CPU)
    sciCore.seed(123)

    val trainIt = DatasetIterator(BATCH_SIZE, MnistDataSupplier(sciCore, train = true, shuffle = false))
    val testIt = DatasetIterator(BATCH_SIZE, MnistDataSupplier(sciCore, train = false, shuffle = false))

    val net = MnistNet(sciCore)

    val optimizer = Sgd(sciCore, LEARNING_RATE, net.parameters())
    for (step in 0 until N_TRAINING_STEPS) {
        sciCore.backend.operationRecorder.scopedRecording {
            val batch = trainIt.next()
            batch.use { x, y ->
                lossValue = net(x)
                    .use { yPred -> yPred.minus(y) }
                    .use { diff -> diff.pow(2f) }
                    .use { diffSquared -> diffSquared.reduceSum(-1) }
                    .use { sum -> sum.divide(BATCH_SIZE.toFloat()) }
                    .use { loss -> optimizer.step(loss); loss.elementAsDouble() }
            }
        }
    }
}
```

## MLP Language Model

```kotlin
class MakeMoreNet(sciCore: SciCore) : IModule {

    private val embedding = sciCore.gaussian(
        DataType.FLOAT32,
        VOCAB_SIZE.toLong(),
        EMBEDDING_SIZE.toLong()
    )
    private val fc1 = Linear(sciCore, DataType.FLOAT32, (EMBEDDING_SIZE * BLOCK_SIZE).toLong(), N_HIDDEN.toLong(), true)
    private val fc2 = Linear(sciCore, DataType.FLOAT32, N_HIDDEN.toLong(), VOCAB_SIZE.toLong(), true)
    private val act = Tanh()

    override fun forward(input: ITensor): ITensor {
        val logits = embedding[input]
            .use { embeddingsForSequence ->
                embeddingsForSequence.getReshapedView(
                    longArrayOf(
                        -1,
                        (EMBEDDING_SIZE * BLOCK_SIZE).toLong()
                    )
                )
            }
            .use { embeddingsForSequenceFlat ->
                fc1(embeddingsForSequenceFlat)
            }
            .use { fc1Output ->
                act(fc1Output)
            }
            .use { fc1WithAct ->
                fc2(fc1WithAct)
            }
        return logits
    }

    override fun subModules(): List<IModule> {
        return listOf(fc1, fc2)
    }

    override fun parameters(): List<ITensor> {
        return listOf(fc1.parameters(), fc2.parameters())
            .flatten()
            .toMutableList()
            .apply { add(embedding) }
    }
}
fun main() {
    val sciCore = SciCore()
    sciCore.setBackend(ISciCore.BackendType.CPU)
    sciCore.seed(123)

    val namesSupplier = NamesDatasetSupplier(sciCore, BLOCK_SIZE, training = true, shuffle = true)
    val trainIt = DatasetIterator(BATCH_SIZE, namesSupplier)

    val net = MakeMoreNet(sciCore)

    val optimizer = Sgd(sciCore, INITIAL_LEARNING_RATE, END_LEARNING_RATE, N_TRAINING_STEPS, net.parameters())
    sciCore.backend.operationRecorder.scopedRecording {
        val batch = trainIt.next()
        batch.use { x, y ->
            lossValue = net(x)
                .use { logits -> logits.exp() }
                .use { counts ->
                    counts.reduceSum(1, true)
                        .use { totalCounts -> counts / totalCounts }
                }
                .use { probs ->
                    sciCore.arange(0, 32, 1, DataType.INT64)
                        .use { idx -> probs[idx, y] }
                }
                .use { probsAssignedToCorrectLabels ->
                    probsAssignedToCorrectLabels.log()
                }
                .use { logProbs ->
                    logProbs.mean()
                }
                .use { logProbsMean ->
                    -logProbsMean
                }.use { loss ->
                    optimizer.step(negativeLogLikelyHood)
                    loss.elementAsDouble()
                }
        }
    }
}
```

## Example of autograd engine

```java
/*
# Neural Network

input = (2, n) # (n_inputs: 2, batch_size: n)

# Layer 1 (Linear, no bias)
w1 = (4, 2) # neurons: 4, input_size: 2
fwd1 = w1 * input
fwd1 = (4, 2) * (2, n)
fwd1 = (4, n)

# Layer 2 (Linear, no bias)
w2 = (1, 4) # neurons: 1, input_size: 4
fwd2 = w2 * fwd1
fwd2 = (1, 4) * (4, n)
fwd2 = (1, n) # (output: 1, batch_size: n)
*/
ITensor act = sciCore.matrix(new float[][]{{1}, {2}}); // (2, n)
ITensor w1 = sciCore.matrix(new float[][]{{3, 4}, {5, 6}, {7, 8}, {9, 10}});
ITensor w2 = sciCore.matrix(new float[][]{{11, 12, 13, 14}});


// Forward pass
ITensor fwd1 = w1.matmul(act);
ITensor fwd2 = w2.matmul(fwd1);

// Automatic backpropagation
IGraph graph = sciCore.getGraphUpTo(fwd2);
graph.requestGradientsFor(w1, w2);
graph.backward();

// Get gradients
ITensor dL_dW2 = graph.getGradient(w2).orElseThrow();
ITensor dL_dW1 = graph.getGradient(w1).orElseThrow();

```

# Same example in torch

```python
import torch

act = torch.tensor([[1], [2]]) # (2, n)
w1 = torch.tensor([[3, 4], [5, 6], [7, 8], [9, 10]], requires_grad=True)
w2 = torch.tensor([[11, 12, 13, 14]], requires_grad=True)

# Forward pass
fwd1 = w1.matmul(act)
fwd2 = w2.matmul(fwd1)

# Automatic backpropagation
fwd2.backward()

# Get gradients
dL_dW2 = w2.grad
dL_dW1 = w1.grad
```