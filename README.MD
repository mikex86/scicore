# SciCore

SciCore is a tiny tensor processing library and autograd engine with a focus on simplicity. It is written in Java,
with some native code for hardware specific optimizations.
At the moment the only fully featured backend is the jvm backend, which is a pure java implementation.
SciCore is very experimental and not ready for production use and primarily serves as a playground to explore the
fundamentals
of deep learning with a tensor processing library that has shallow levels of abstraction.

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
ITensor act = sciCore.matrix(new float[][]{{1}, {2}});
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

## Same example in torch

```python
import torch

act = torch.tensor([[1], [2]])
w1 = torch.tensor([[3, 4], [5, 6], [7, 8], [9, 10]], requires_grad=True)
w2 = torch.tensor([[11, 12, 13, 14]], requires_grad=True)

// Forward pass
fwd1 = w1.matmul(act)
fwd2 = w2.matmul(fwd1)

// Automatic backpropagation
fwd2.backward()

// Get gradients
dL_dW2 = w2.grad
dL_dW1 = w1.grad
```

## Example of the Neural Network library

```java
class BobNet implements IModule {
  
    private final Sigmoid act = new Sigmoid();
  
    private final Linear f1 = new Linear(sciCore, DataType.FLOAT32, 1, 1, true);
  
    private final Linear f2 = new Linear(sciCore, DataType.FLOAT32, 1, 1, true);
  
    public ITensor forward(ITensor input) {
        ITensor out = f1.forward(input);
        out = act.forward(out);
        out = f2.forward(out);
        return out;
    }
  
    @Override
    public List<ITensor> parameters() {
        return collectParameters(f1, f2);
    }
}

SciCore sciCore = new SciCore();
sciCore.setBackend(SciCore.BackendType.JVM);
                
DatasetIterator dataIt = getData(batchSize);
IOptimizer optimizer = new Sgd(sciCore, 0.5f, bobNet.parameters(), true, 1e-6f);
for (int step = 0; step < nSteps; step++) {
    sciCore.getBackend().getOperationRecorder().resetRecording();
    Pair<ITensor, ITensor> next = dataIt.next();
    ITensor X = next.getFirst();
    ITensor Y = next.getSecond();
    ITensor YPred = bobNet.forward(X);
    ITensor loss = (YPred.minus(Y)).pow(2).reduceSum(0).divided(batchSize);
    IGraph graph = sciCore.getGraphUpTo(loss);
    optimizer.step(graph);
}
```
