# Goal

The goal of this thesis is to lay the theoretical foundation for implementing a conversational AI system from first principles.
This conversational AI system requested by the client (Higher Technical College St. Pölten) is aimed to be used in the context of a chatbot for the purpose of providing information about the school.
To explore the low-level mathematical foundations of this problem to a deeper extent, a library for accelerating the
tensor processing operations commonplace in deep learning was developed. Note that this library will not be used in the final project,
but rather a tool to help explore the problem, as the writing a production-ready and fully featured tensor processing library requires a lot of
time and resources, usually only available to large organisations such as Google, Facebook, and Microsoft.
The library will be open source and available on GitHub under the MIT license. The library is written in Java, while some native code is used for hardware-specific acceleration,
such as the Intel MKL library for matrix multiplication on x86 CPUs, and the Apple Accelerate framework on Apple Silicon CPUs, as well as the CUDA toolkit for NVIDIA GPUs.
The library is designed with a focus on readability with shallow levels of abstraction in order to not obfuscate the underlying mathematics.
Besides these vendor-specific libraries, the library is written in pure Java and does not use any other third-party libraries.

# Overview

In the course of this thesis, we will express the problem of conversational AI as a mathematical problem.
Concepts will be introduced from first principles and the layers of abstraction traversed from the low-level mathematical foundations to the high-level concept that is a conversational AI system.

To start, a definition for 'learning' in the machine learning context will be layed out, and the concept of a neural network will be introduced as an example of a system learning in such fashion.
The concept of a tensor will be introduced, which is a generalisation of a vector and a matrix and is the fundamental data structure in deep learning.
While neural networks in their raw form of sufficient complexity are considered universal function approximators, architectures with intent to speed up the learning process in light of the task-specific problem structure are introduced.
The concept of a recurrent neural network will be introduced as an example of such an architecture, as a way to handle sequential data, while also focusing on their shortcomings.
The concept of a transformer will be introduced as a way to overcome the shortcomings of recurrent neural networks.
Lastly, we will introduce the concept of language models, which are a special case of transformers, as well the concept of a chatbot, which is a special case of language models.

<div style="page-break-after: always;"></div>


# Definition of 'Learning'

Learning is a concept that is used in many different contexts, but in the context of machine learning, it is defined as the process of approximating an unknown function that maps an input to an output.
This approximation will be derived from a set of examples called the training data, where each example $(X, Y)$ is a pair of an input $X$, also called the feature vector, and an output $Y$, also referred to as the label.
The goal of the learning process is to find a function $f(x)$, also known as the model $m(x)$, that maps the input $X$ to the output $Y$, but specifically a function that will be able to map inputs $X$ to outputs $Y$ in a generalizing fashion, meaning that the
function will be able to map inputs $X$ to the correct outputs $Y$ that were not part of the set of examples used to derive the function.
Thus, from now on, we will refer to the term generalization as the predictive capability of the model $m(x)$ beyond the set of examples used to derive the function as well as the predictive capability of the function $m(x)$
for inputs $X$ outside the proximity of the set of examples used to derive the function. While for problems of low complexity, "proximity" in this context can be equated to distance metrics such as the Euclidean distance,
an arbitrarily complex problem $P(X)$ has no inherent distance metric, as whether two inputs $(X1, X2)$ are considered similar in the context of the problem is determined by the nature of the problem itself.
Thus, the concept of "proximity" shall not be a mathematical term, but rather a colloquial term for what a reasonable interpeter of the problem would consider to be similar inputs.

# Tensors

## Definition of a Tensor
A tensor is a generalisation of a vector and a matrix, and is the fundamental data structure in deep learning.
In its most general case a tensor $T$ of rank $n$ can be denoted as $T \in \mathbb{R}^{n_1 \times n_2 \times \dots \times n_n}$, where $n_i$ is the length of the $i$-th dimension of the tensor.
A tensor is a special case of multi-dimensional array of numbers or other primitives where for all elements of dimension $n$, the length is equivalent.
Thus the following condition is impossible in the constraints of a tensor: `array[n][0].length != array[n][1].length`.


### Shape and Rank
Given the constraint of a tensor compared to general multi-dimensional arrays, the shape of the tensor can be defined as a tuple of integers, where each element represents the length of the corresponding dimension.
Eg. the shape of a tensor with dimensions of length 2, 3 and 4 is $(2, 3, 4)$, where the first dimension is the most significant dimension and the last dimension is the least significant dimension, meaning that $4$ is a dimension of scalars.
The number of indices required to access a single scalar element from a given tensor is called the rank of the tensor, and is equivalent to the number of dimensions of the tensor. The rank of a tensor is not to be confused with the rank of a matrix in linear algebra, where the rank of a matrix is the number of linearly independent columns or rows.
Tensors of rank `0` are scalar tensors, vectors can be represented as tensors of rank `1` and matrices can be represented as tensors of rank `2`.


### Denoting a tensor with a specific multiset of values
A scalar tensor will be denoted indistinguishably as a numeric/primitive value, such as $3.14$, $-2$ or `true`.
A tensor of rank 1 will be denoted as a vector, such as $\begin{bmatrix} 1 & 2 & 3 \end{bmatrix}$, while tensors of rank two will be denoted as matrices, such as $\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}$.
Tensors of higher ranks will only be denoted in array notation, such as `[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]`.

### Indexing
Indexing into a tensor will be denoted as $T_{i_1i_2 \space \dots \space i_n }$ where n is the number of dimensions to index into. When all dimensions are indexed into, meaning $n$ is equal to the rank of the tensor, the result is a scalar value, otherwise the result is another tensor of rank $n - k$ where $k$ is the number of dimensions indexed into.

## Implementing the Tensor data structure
While java has built-in support for multidimensional arrays, they are unsuitable for purposes of deep learning in a multitude of ways.
Firstly, given that java is a strongly typed language, the type of the elements of the array and the dimensionality of the array must be specified explicitly, making arrays unwieldy given that functions processing tensors should be able to operate largely independenly of the data type and dimensionality of the input tensor.

Secondly, accelerating tensor operation with hardware-specific optimizations is made difficult by the inability to control how tensor data is stored on a low level when using a language-intrinsic feature, resulting in unnecessary copies of data when operations are performed via an accelerator that is not the CPU, or even outside the JVM, which is an unnecessarily costly operation.

Thus SciCore defines its own `ITensor` interface that is implemented in different backends depending on the specific hardware acceleration that is requested.
Although each implementation differs ever so slightly in the way it is implemented on the given hardware, all implementations follow a similar pattern of storing the contents of the tensor in memory.

### Tensor Data Storage and shape-general Indexing
The data of a tensor is stored in a one-dimensional array of primitive values.

This one-dimensional array is then indexed via what we will now refer to as a "flat index". The flat index is the index of the element in the one-dimensional array that corresponds to the element in the tensor that is indexed by the multi-dimensional index. To calculate the flat index, we will be introducing a new concept called "strides".

#### Strides
The strides of a tensor is a tuple of integers, where each element at index $n$ represents the number of elements that must be skipped in the one-dimensional array to get to the next element in the corresponding dimension $n$.
Eg. the strides of a tensor with shape $(2, 3, 4)$ is $(12, 4, 1)$.
On the third dimension that is the scalar dimension of $4$ elements, to iterate to the next element, we skip $1$ element in the flat array, on the second dimension of $3$ elements, we skip $4$ elements in the flat array to get to the next element, and on the first dimension of $2$ elements, we skip $12$ elements in the flat array to get to the next element.

The following graphic illustrates the strides of a tensor with the shape $(3, 4)$ and the strides $(4, 1)$.
Note how iterating 4 elements in the flat array corresponds to iterating 1 element in the first dimension,
and iterating 1 element in the flat array corresponds to iterating 1 element in the second (scalar-level) dimension.

![Strides](figures/strides_visualization.svg)

Strides are derived from the shape of the tensor, and are calculated as follows:

```java
public static long[] makeStrides(long[] shape) {        
    long[] strides = new long[shape.length];
    strides[strides.length - 1] = 1;
    for (int dim = shape.length - 2; dim >= 0; dim--){
        strides[dim] = strides[dim + 1] * shape[dim + 1];
    }
    return strides;
}
```
We start by creating an array of strides of the same length as the shape of the tensor, and set the last element of the strides array to 1.
Then we iterate over the dimensions of the tensor in reverse order, and for each dimension, we set the stride of the current dimension equal to the stride of the last dimension multiplied by the length of the last dimension.

#### Flat Indexing
Given the strides of a tensor, we can now calculate the flat index of an element in the tensor given its multi-dimensional index. This is acomplished by multiplying $strides_n$ * $index_n$ for all $n$ in the range $0$ to $rank - 1$ and summing the results.

```java
public static long getFlatIndex(long[] index, long[] strides) {
    long flatIndex = 0;
    for (int dim = 0; dim < index.length; dim++) {
        flatIndex += index[dim] * strides[dim];
    }
    return flatIndex;
}
```

#### Access Elements of Tensors
To access scalar values in the tensor, the multi-dimensional index supplied to the access method must be of the same length as the rank of the tensor.
The method will then calculate the flat index of the element in the tensor that corresponds to the supplied multi-dimensional index, and return the value at that index in the one-dimensional array.
When using a type-agnostic buffer, we must multiply the flat index by the size of the primitive type of the tensor to get the final byte offset into the buffer.

To show how this is accomplished in SciCore, the following excerpts from the source code will be shown:

```java
public class GenCPUTensor {
    ...
    @Override
    public int getInt(long[] indices) {
        long index = ShapeUtils.getFlatIndex(indices, this.strides);
        return this.dataContainer.getInt32Flat(index);
    }
    ...
}
```
```java
public class GenCPUDataContainer {
    ...
    private final DirectMemoryHandle memoryHandle;
    ...
        long nBytes = dataType.getSizeOf(nElements);
        this.memoryHandle = memoryManager.calloc(nBytes);
    ...
    public int getInt32Flat(long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 4;
        return MemoryUtil.memGetInt(finalPtr);
    }
    ...
}
```

To perform a write instead of a read, the same index calculation is performed, and the value is written to the calculated index in the one-dimensional array.

```java
public class GenCPUTensor {
    ...
    @Override
    public void setInt(long[] indices, int value) {
        long index = ShapeUtils.getFlatIndex(indices, this.strides);
        this.dataContainer.setInt32Flat(index, value);
    }
    ...
}
```

```java
public class GenCPUDataContainer {
    ...
    public void setInt32Flat(int value, long flatIndex) {
        long finalPtr = memoryHandle.getNativePtr() + flatIndex * 4;
        MemoryUtil.memPutInt(finalPtr, value);
    }
    ...
```

## Mathematical operations on tensors

To perform mathematical operations on tensors, we have to implement a well-defined operator that takes one or more tensors as operands and returns a tensor as a result.
Operators are categorized by the number of operands they accept into unary, binary, etc. operators.

### Unary operators
For unary operators, the result tensor is of the same shape as the operand tensor, and the result of the operation can thus be computed by iterating over the elements of the operand tensor in an element-wise fashion and performing the operation on each element, while respecting the strides of the operand tensor.

As an example, we take the `exp` operator, which takes a tensor as an operand and returns a tensor with the same shape as the operand, where each element is the $e^x$ of the corresponding element in the operand tensor.

```java
public class JvmExpOp ... {
    ...
    public ITensor perform(Graph.IOperationContext ctx, ITensor input) {
        long[] shape = input.getShape();
        long[] strides = input.getStrides();
        long nElements = ShapeUtils.getNumElements(shape);
        DataType dataType = input.getDataType();
        ITensor result = backend.createTensor(dataType, shape);
        for (long i = 0; i < nElements; i++) {
            double value = input.getAsDoubleFlat(i);
            result.setByDoubleFlat(Math.exp(value), i);
        }
        result = result.getReshapedView(shape, strides);
        ...
        return result;
    }
    ...
}
```
Note that the strides of the operand tensor are not used to iterate over the elements of the tensor, keeping the implementation simple and fast.
The result is then reinterpreted with the strides of the operand tensor, so that the result tensor has the same strides as the operand tensor.
This way, element order across dimensions is preserved. This comes at the cost of not reordering the elements in the underlying storage of the tensor to a potentially more cache-friendly order, which could slow down subsequent operations performed on the resulting tensor.

An alternative solution would be to iterate over elements of the operand tensor in the order of the strides, and thus reordering the elements in the result tensor in the process. Note that the cache-friendliness of the result tensor comes at the cost of the operation itself being cache-unfriendly, when application of strides results in sequential or non-spatially contiguous access to the elements of the operand tensor in the underlying storage memory.


### Binary operators
To perform binary operations on tensors, such as addition, multiplication, etc., we need a robust way to handle the operands differing in shape.
In the trivial case, where both operands have the same shape, the operation can be performed element-wise, and the result is a tensor with the same shape as the operands.
However, when the operands differ in shape, we need a way to handle the mismatch.
Broadcasting is a concept that is used to handle this case.

#### Broadcasting
Broadcasting is a concept that allows tensors of different shapes to be used in binary operations.
To motivate the need for broadcasting, consider the following example:

```java
ITensor a = sciCore.array(new float[]{1, 2, 3, 4, 5});
ITensor b = sciCore.scalar(2f);

ITensor c = a.multiply(b);
```

The result of the multiplication $C = A * B$ is a tensor with the same shape as $A$, where each element is the product of the corresponding element in $A$ and $B$.
But note that the shape of $B$ is a scalar, while the shape of $A$ is a vector.
Figuratively speaking, we can imagine the scalar $B$ being stretched to the shape of $A$ to match the shape of $A$.

<p align="center">
    <img src="figures/broadcasting_scalar_stretch.svg" alt="Neural Network" width="450vh"/>
</p>
While we could define tensor by scalar multiplication as its own case - which is certanly something to consider for optimization purposes, it is more convenient to view it as a special case of tensor by tensor multiplication under application of the broadcasting rules.

In the general case, the broadcasting rules are as follows:
The dimensions of the shape of both operands are compared from the last dimension (the rightmost dimension) to the first dimension (the leftmost dimension).
Dimensions are compared in pairs, and the following rules are applied:
1. If both dimensions are equal, the dimension in the output shape stays the same.
2. If one of the dimensions is 1, the dimension in the output shape is the same as the dimension in the other operand respectively.
3. If no pair can be constructed because one of the operands has lower rank than the other, the lower ranked operand is prepended with dimensions of size 1 until the ranks match.

The following code snippet implements the broadcasting rules to compute the shape of the resulting tensor of a binary operation:
```java
public static long[] broadcastShapes(long[] shapeA, long[] shapeB) {
    long[] broadcastShape = new long[shapeA.length];
    for (int i = 0; i < shapeA.length; i++) {
        long elementA = shapeA[shapeA.length - 1 - i];
        long elementB = i < shapeB.length ? shapeB[shapeB.length - 1 - i] : 1;
        long dimSize;
        if (elementA == elementB || elementB == 1) {
            dimSize = elementA;
        } else if (elementA == 1) {
            dimSize = elementB;
        } else {
            throw new IllegalArgumentException(
                    "Shapes are not broadcast-able: shapeA: " 
                    + ShapeUtils.toString(shapeA) +
                    ", shapeB: " + ShapeUtils.toString(shapeB)
            );
        }
        broadcastShape[broadcastShape.length - 1 - i] = dimSize;
    }
    return broadcastShape;
}
```
When a dimension is stretched to match the shape of the other operand, the elements of said dimensions are repeated to fill the dimension.
Note that if we were to actually make copies of the elements, the operation would be very memory wasteful, as the same information would be stored multiple times.
As can be seen in the above figure, the tensor $B$ is expanded to match the shape of $A$ by repeating the elements of $B$ along the first dimension, which would involve making
four additional copies of the element $2$. While not fatal for small tensors, this would be a very wasteful operation for large tensors.
Instead, we can "virtually extend" the shape of $B$ to the desired shape.

##### Virtually extended Tensors
Virtually extended tensors can be implemented in a multitude of ways.
Firstly, we can set the stride of a stretched dimension to zero. This way, advancing to the next element in the stretched dimension will result in the same element being accessed.
However, n-dimensional general index computation is a comparitively expensive operation, and we would like to avoid it as much as possible.
Thus we want to derive specialized solutions for indexing problems whenever possible.
We will thus introduce the terminology of "index constraining", meaning to restrict possible indices to a subset of the full dimension space.
In the case of stretched dimensions, we can constrain the indices of a stretched dimension to an index range of size 1.
The following code snippet implements the index calculation to map the `outputIndex`, which is a dimensional index into the result tensor of the operation, to the corresponding index in the operand tensor, which should be employed for element-whise operation.
```cpp
size_t getFlatIndexConstrained(const size_t *outputIndex,
                                const size_t *shape, const size_t *strides,
                                size_t nDims, size_t nDimsOut) {
    assert(nDims <= nDimsOut);
    size_t nNewDims = nDimsOut - nDims;
    size_t flatIndex = 0;
    for (size_t dim = 0; dim < nDims; dim++) {
        size_t stride = strides[dim];
        flatIndex += (outputIndex[dim + nNewDims] % shape[dim]) * stride;
    }
    return flatIndex;
}
```

Eg. in the binary multiplication operation, the method is used as follows:
```cpp
template<typename A, typename B, typename C>
void tblas_multiply(const A *a, const B *b, C *c,
                    size_t *shapeA, size_t *stridesA, size_t nDimsA,
                    size_t *shapeB, size_t *stridesB, size_t nDimsB,
                    size_t *shapeC, size_t *stridesC, size_t nDimsC) {
    auto *outputIndex = new size_t[nDimsC];
    memset(outputIndex, 0, sizeof(size_t) * nDimsC);

    size_t cIndexFlat = 0;
    do {
        size_t aIndexFlat = getFlatIndexConstrained(
                                outputIndex,
                                shapeA, stridesA, nDimsA,
                                nDimsC
                            );
        size_t bIndexFlat = getFlatIndexConstrained(
                                outputIndex,
                                shapeB, `stridesB, nDimsB,
                                nDimsC
                            );
        c[cIndexFlat] = a[aIndexFlat] * b[bIndexFlat];
        cIndexFlat++;
    } while (incrementIndex(outputIndex, shapeC, nDimsC));
    delete[] outputIndex;
}
```

# Neural Networks
Neural networks are a class of machine learning models inspired by the human brain.
The human brain is composed of neurons, which are interconnected and communicate with each other via electrical signals. The signals are sent from one neuron to another via synapses, which are the connections between neurons. Wether a neuron fires or not is determined by the strengths of the signals it receives from the neurons it is connected to.

## Artificial Neurons
We can create a primitive model of a neuron by defining a function that takes a set of inputs and returns a single output. The most common model model of a neuron sums these inputs and applies a non-linear function to the sum, called the activation function $g(x)$. Generally, any non-linear function can be used as an activation function, but in practice functions are chosen that have desirable properties, such as how easy it is to calculate the derivative of the function, or how it transforms the real number line $\mathbb{R}$.
The output of such a neuron is called the activation of the neuron, and is denoted as $a$.
Generally, the activation of a neuron is defined as:
$$
a = g(\sum_{i=1}^{n} w_i x_i + b)
$$
where $w_i$ is the weight of the $i$-th input, $x_i$ is the $i$-th input, and $b$ is the bias of the neuron.
Each input can be thought of as a connection from another neuron, and the weight scales the influence of that connection to the output activation of the neuron. The bias can be thought of as a constant input to the neuron used to shift the activation of the neuron.
The weights $w$ and the bias $b$ are the parameters of the neuron and their specific values determine the behavior of the neuron and thus how it responds to inputs. The weights and bias are adjusted during training.

## Artificial Neural Networks
Artificial neural networks are composed of multiple such neurons, generally organized in layers in a fully-connected manner, meaning that each neuron in one layer is connected to every neuron in the next layer.
Generally, artificial neural networks consist of an input layer, one or more hidden layers, and an output layer.
The input layer is composed of neurons that take in the input data that the network should work with, and the output layer is composed of neurons hold the prediction and thus the output of the network. The hidden layers are the set of neurons that perform the actual computation of the network.

<p align="center">
    <img src="figures/neural_network.svg" alt="Neural Network" width="300vh"/>
</p>

The number of hidden layers and the number of neurons in each layer are hyperparameters of the network, and are chosen based on the problem at hand.
Generally, the more complex the problem, the more hidden layers and neurons are needed to solve it. Given that with each additional layer, a non-linearity is added to the activations, the number of hidden layers thus determines the complexity of the function that the network can approximate. Eg. a network with one hidden layer can only approximate linear functions, but would fail to approximate a parabola. For more complex task, there is no obvious answer on what the appropriate number of hidden layers and neurons is, and it is often determined by trial and error.

The general activation for a multi-layer neural network is defined as:

$$
a_j^{[l]} = g^{[l]}(\sum_{k=1}^{n^{[l-1]}} w_{jk}^{[l]} a_k^{[l-1]} + b_j^{[l]})
$$

where $a_j^{[l]}$ is the activation of the $l$-th layer of the $j$-th neuron, $g^{[l]}$ is the activation function of the $l$-th layer, $w^{[l]}$ is the weight matrix of the $l$-th layer, $b_j^{[l]}$ is the bias of the 
$j$-th neuron of the $l$-th layer, and $n^{[l-1]}$ is the number of neurons in the previous layer.
The weight matrix is organized such that the weight at $w_{jk}$ is the weight of the connection from the $k$-th neuron of the previous layer to the $j$-th neuron of the current layer.


An astude observer might notice that this is mathematcally equivalent to the definition of a matrix multiplication, and indeed, the activation of a layer can be calculated as a matrix multiplication/dot product of the activations of the previous layer and the weight matrix of the current layer, plus the bias vector of the current layer:

$$
a^{[l]} = g^{[l]}(W^{[l]}\times a^{[l-1]} + b^{[l]})
$$


## ANNs in Deep Learning Frameworks

In modern deep learning frameworks, neural networks are typically represented as so-called "modules", which implement the so-called "forward pass", which is a function that takes the input data and propagates it through the network and returns the final output of the network. The deeplearning framework usually also provides modules for common neural network layers, such as a fully-connected layer. This allows the user to easily build complex neural networks by combining these modules by passing the output of one module as the input to another module in the forward pass.
A fully-connected layer is usually refered to as a `Dense` or `Linear` layer.
The `Linear` layer is such a module, and thus it implements the `forward` method, which takes the input data and returns the output of the layer. The `forward` method of the `Linear` layer is a simple matrix multiplication, equivalent to the mathematical definition above. The weights and bias are initialized uniformly between $\mathcal{U}(-\sqrt{k}, \sqrt{k})$ where $k=\frac{n}{in\_features}$. This is the so-called Xavier initialization, which a "commonly used heuristic" to initialize weights in neural networks. *(Xavier Glorot, Yoshua Bengio - 2010)* It aims to keep the variance of the activations of the layer constant, which is desirable as it prevents the activations from exploding or vanishing. But note that even though this is commonly used heuristic, it is only that - a heuristic and not a mathematically proven method. In fact, simple testing reveals, that only after a few layers the variance of the activations can shrink drastically, to the point where lack floating point precision destabilizes gradient descent and the network fails to train. We will discuss gradient descent in further detail in a later chapter.

Note that the usage of a bias is optional and can be disabled by setting the `bias` parameter to `false`.
The following snippet shows releveant parts of the implementation of the `Linear` module.

```java
public class Linear implements IModule {
    ...
    private final ITensor weights;

    @Nullable
    private final ITensor bias;

    ...

        float k = (float) (1.0 / Math.sqrt(inputSize));
        this.weights = sciCore
            .uniform(dataType, outputSize, inputSize)
            .multiply(2 * k)
            .minus(k);
        if (useBias) {
            this.bias = sciCore
                .uniform(dataType, outputSize)
                .multiply(2 * k)
                .minus(k);
        } else {
            this.bias = null;
        }

    ...

    @Override
    public ITensor forward(ITensor input) {
        ...
        ITensor x = input.matmul(weights.transpose());
        if (bias != null) {
            x = x.plus(bias);
        }
        return x;
    }
    ...
}
```

In SciCore, the `Linear` layer can be used as follows:

```java
class BobNet implements IModule {

    private final Sigmoid act = new Sigmoid();

    private final Linear fc1 = new Linear(sciCore, DataType.FLOAT32, 4, 5, true);
    private final Linear fc2 = new Linear(sciCore, DataType.FLOAT32, 5, 5, true);
    private final Linear fc3 = new Linear(sciCore, DataType.FLOAT32, 5, 3, true);

    public Tensor forward(Tensor input) {
        Tensor h = fc1.forward(input);
        h = act.forward(out);
        h = fc2.forward(out);
        h = act.forward(out);
        h = fc3.forward(out);
        return h;
    }

    @Override
    public List<ITensor> parameters() {
        return collectParameters(fc1, fc2, fc3);
    }
}
```

The `Linear` layer is initialized with the `sciCore` instance, the data type, the number of input features, the number of output features, and a boolean indicating whether the layer should use a bias or not.

Note how a multi-layer neural network is simply a composition of `Linear` layers in combination with an activation function, such as `Sigmoid` in the example above. 

The parameters method returns all the trainable parameters of the network, which are in this case the weights and bias of each layer. This is important for the training of the network, as the parameters need to be updated during the training process.

### Matrixmultiplication in SciCore
Given that modules such as `Linear` rely on fast implementations of common operations such as matrix multiplication, these core operations are optimized in hardware-specific backends.
Eg. for the CUDA backend, the matrix multiplication is implemented using the cuBLAS library, which implements common BLAS (Basic Linear Algebra Subprograms) operations such as matrix multiplication in an efficient manner on the GPU.
The following code snippet shows how matrix multiplication is implemented in on the CUDA backend:

```java
public class CudaMatmulOp implements IDifferentiableBinaryOperation {
    ...
    @Override
    public ITensor perform(
        Graph.IOperationContext ctx,
        ITensor a, ITensor b
    ) {
        ...
        cublasCheck(cublasGemmEx_new(
                    CudaBackend.getCublasHandle(),
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k,
                    Pointer.to(factor),
                    bMemoryHandle.getDevicePointer(),
                    CUDA_R_32F,
                    n,
                    aMemoryHandle.getDevicePointer(),
                    CUDA_R_32F,
                    k,
                    Pointer.to(factor),
                    resultMemoryHandle.getDevicePointer(),
                    CUDA_R_32F,
                    n,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DFALT_TENSOR_OP
            ));
        ...
    }
    ...
}
```

For data type combinations unsupported by cuBlas, the matrix multiplication is implemented using a custom CUDA kernel. Note that this kernel is a naive implementation that lacks many optimizations that are present in the highly-efficient cuBLAS implementation.

```cpp
template <typename A, typename B, typename C>
KERNEL_TEMPLATE void matmul(A *a, B *b, C *c, size_t m, size_t n, size_t k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || j >= n) {
      return;
    }
    C sum = 0;
    for (int l = 0; l < k; l++) {
      sum += a[i * k + l] * b[l * n + j];
    }
    c[i * n + j] = sum;
}
```

# Training
Defining the structure of the function with which to approximate a given problem $P(X)$ is only half of the story. The trainable parameters of such a function must be chosen in such a way that the function approximates $P(X)$ as closely as possible. The process of iteratively updating the parameters of the model to improve the approximation is called training.

## Loss function
For this purpose, we introduce a metric called "loss" which shall represent the performance of our network on the specified problem in a single scalar value. 
In general the loss can be thought of as the divergence between the output of the network and the desired output. Thus, a low loss is desirable. There are many possible methods to compute the loss, and the choice of loss function is highly problem-dependent. A very common loss function, but also very simple loss function, is the mean squared error (MSE) loss, which is defined as follows:

$$
J(\hat{y}_i, y_i) = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
$$
where $y_i$ is the desired output and $\hat{y}_i$ is the output of the network for the $i$-th example and N is the number of examples in the training dataset. The loss function is also sometimes denoted as $L$.

## Gradient descent
Given that the loss function $J(\hat{y}_i, y_i)$ is a function of $\hat{y}_i$, which in turn is a function of the model $m(x, W)$, we can for a specific example $x$ caluclate the partial derivative of the loss function with respect to the parameters $W$ of the model. This is called the gradient of the loss function with respect to the parameters of the model.
Given the gradient of the loss function with respect to a specific parameter $W_i$, we can update the parameter $W_i$ in the direction of the negative gradient, which will decrease the loss function. This is called gradient descent:
$$
W_i = W_i - \alpha \frac{\partial J}{\partial W_i}
$$
where $\alpha$ is the learning rate, which determines the size of the step in the direction of the negative gradient. While increasing the learning rate will result in larger steps, the training process may become unstable and the loss may not converge to a local minimum. Due to the non-convex nature of the loss function, the loss may not converge to a global minimum, but only to a local minimum. While this might seem to be a problem at first, converging to the absolute global minimum is not always desirable and in fact may result in a model that fails to generalize beyond examples found in the training data.

## Backpropagation
Backpropagation is an algorithm to find the gradients to all parameters $W_1, W_2, ... W_n$ of the model $m(W)$ that we want to optimize. The algorithm is simply an implementation of the rules of calculus, and in particular it provides an intuitive interpretation of the chain rule. The chain rule states that the following:
$$
\frac{\partial dL}{\partial dx} = \frac{\partial dL}{\partial dz} \frac{\partial dz}{\partial dx}
$$
where $L$ is the function we are partially differentiating (eg. the loss function), $z$ is the output of an function $f(x)$ and $x$ is the variable we are partially differentiating with respect to.
An interpretation of this formula common in the field of deep learning is that the chain rule simply states that the following:
$$
global gradient = upstream gradient * local gradient
$$
In this interpretation, the function to differentiate is represented as a directed acyclic graph (DAG) of individual differentiable operations, to which the chain rule is then recursively applied. In this interpretation our loss function $L$ can be viewed as the root node of the graph of operations and thus the origin, where the recursive differentiation alogirithm starts from.

The following figure visualizes this interpretation of the chain rule applied to a well-defined binary operator:

![upstream and local gradients](figures/upstream-local-gradient.svg)

In this figure, we see a well-defined differentiable binary operator $f(x, y)$ which computes an output $z$. The output $z$ is an intermediary or the final value of the forward pass. To differentiate this operation, wee need the gradients "up until this operation" - the upstream gradients $\frac{\partial L}{\partial z}$. Then we proceed by computing the local gradients for both inputs $x$ and $y$. The local gradients only tell us how the output $z$ is influenced by the input $x$ or $y$. The upstream gradients tell us how the output $z$ affects the loss function $L$. The product of the local and upstream gradients is the "global gradient" of the function $L$ with respect to the input $x$ or $y$. Note that $x$ and $y$ can be functions themselves. In this case the gradients we computed are referred to as "downstream gradients" and become the "upstream gradients" for the next operation in the chain that computed the given input variable. 

Note that normally we will determine whether differentiating in respect to a given input of the operation is even necessary given what paramters we want to differentiate with respect to.
Software capable of differentiating such arbitrary graphs of operations is referred to as "Autograd engines" and are the backbone of modern deep learning frameworks.

### Scalar-level Autograd
We will now explore implementing a simple scalar-based autograd engine. Note that this simple autograd engine is not part of SciCore, as this approach is not suitable for efficient differentiation of large neural networks.
Later, we will explore autograd where the "atoms" of differentiations are not the individual scalars, but rather the tensors that are the inputs to tensor-based operations.
The code for this section can be found on GitHub: https://github.com/mikex86/scalargrad

First, we will define a simple `Value` class to represent the scalar values that we will use as computational atoms.

```java
public class Value {
    private double v;
    private double grad;
    ...
    public Value(double v) {
        this.v = v;
        this.grad = 0;
    }
    ...
}
```
Now we will define a simple `Operation` interface that will represent the operations that we will perform on the `Value` objects. The `Operation` interface will have a method `perform` that will compute the output of the operation given its inputs and a method `backward` that will compute the gradients of the operation with respect to its inputs given the output value that the operation computed in the forward pass. The gradient of said value is the upstream gradient $\frac{\partial L}{\partial z}$.

```java
public interface Operation {

    Value perform(List<Value> inputs);

    void backward(Value output, List<Value> inputs);

}
```

The first operation we will implement is the addition operation. The addition operation is defined as follows:

```java
public class PlusOp implements BinaryOperation {

    @Override
    public Value perform(Value a, Value b) {
        return new Value(a.getValue() + b.getValue());
    }

    ...
}
```


The second operation we will implement is the multiplication operation. This operation will take two inputs and is thus a BinaryOperation, which redefines the `perform`
method to take two `Value` inputs.

```java
public class MultiplyOp implements BinaryOperation {

    @Override
    public Value perform(Value a, Value b) {
        return new Value(a.getValue() * b.getValue());
    }

    ...
}
```

The third operation we will implement is the pow operation.

```java
public class PowOp implements BinaryOperation {

    @Override
    public Value perform(Value a, Value b) {
        return new Value(Math.pow(a.getValue(), b.getValue()));
    }

    ...
}
```

Now we will implement the `Graph` class that will represent the computational graph of operations. The graph will consist of `Node` objects, which reference each other in a directed acyclic fashion. `Node` objects will store `Value` objects, which is the value of the node. This can either be the output of an `Operation`, or simply a declared value.

```java
public class Graph {

    private final Node rootNode;

    public Graph(Node rootNode) {
        this.rootNode = rootNode;
    }
    ...
}
```

There are two types of `Node` objects: `OperationNode` and `ValueDeclarationNode`. The `OperationNode` will store an `Operation` object, the list of input `Node` objects, and the output `Value` object. The `ValueDeclarationNode` will only store a `Value` object.

```java
public static class OperationNode extends Node {

    private final Operation operation;
    private final List<Node> inputNodes;

    public OperationNode(Operation operation, List<Node> inputNodes, Value output) {
        super(output);
        this.operation = operation;
        this.inputNodes = inputNodes;
    }
    ...
}
```

```java
public static class ValueDeclarationNode extends Node {

    public ValueDeclarationNode(Value value) {
        super(value);
    }
    ...
}
```

We will now define a mechanism to record `Operations` on the fly into a graph.
This `GraphRecorder` will be a singleton object that will record all operations performed on `Value` objects. The `GraphRecorder` will store the a mapping from `Value` objects to their associated `Node` objects in the graph. 

```java
public class GraphRecorder {

    private final Map<Value, Graph.Node> valueToNodeMap = new HashMap<>();

    public Value recordOperation(Operation operation, List<Value> inputs) {
        Value output = operation.perform(inputs);

        List<Graph.Node> inputNodes = new ArrayList<>();

        // look up which operations computed the values that we see as inputs
        for (Value inputValue : inputs) {
            Graph.Node inputNode = valueToNodeMap.get(inputValue);
            if (inputNode == null) {
                // if we don't have a node for this value,
                // it means that it is a value declaration
                inputNode = new Graph.ValueDeclarationNode(inputValue);
            }
            inputNodes.add(inputNode);
        }

        Graph.Node node = new Graph.OperationNode(operation, inputNodes, output);
        valueToNodeMap.put(output, node);

        return output;
    }

    public Graph endRecording(Value rootValue) {
        return new Graph(
            Optional.ofNullable(valueToNodeMap.get(rootValue))
                .orElseThrow(() ->
                    new IllegalArgumentException("Value not contained in graph!")
                )
            );
    }
}
```

The `GraphRecorder` will be used in the `Value` class to record operations performed on said `Value` objects.

```java
public class Value {
    ...
    public Value multiply(Value b) {
        return graphRecorder.recordOperation(new MultiplyOp(), List.of(this, b));
    }

    public Value pow(Value value) {
        return graphRecorder.recordOperation(new PowerOp(), List.of(this, value));
    }
    ...
}
```

These handy methods will allow us to write code of the following fashion:
```java
Value a = new Value(2);
Value b = new Value(4);

Value c = a.multiply(b);
```

We will now implement the `backward` method in the `Graph` class. This method will perform the backward pass of the graph, computing the gradients of all the `Value` objects in the graph. Note that normally one would only want to compute the gradients of a subset of the `Value` objects in the graph, where it is explicitly required, which is a feature that we will not implement here for the sake of simplicity.

In the following code snippet, we apply the chain rule by first computing the downstream gradients of the input nodes of the currently traversed operation and ascending the graph to all nodes the operation depends on.
Computed downstream gradients will become the upstream gradients of the next operation during traversal.
Gradients are always accumulated, so that when multiple operation nodes depend on the same value, all contributions are taken into account.
Note that we traverse in such an order we only move on to the next operation in the graph when all operations that could influence the gradient of a particular operation have been traversed. This ensures we do not move on to the next operation with pre-maturely computed upstream gradients of downstream operations. This for example happens in a softmax operation, where a particular result is used at different depths of the graph, resulting in two possible paths to all upstream operations.

```java
public class Graph {
    ...
    private void backward(Node node) {
        Deque<Node> topology = new LinkedList<>();
        Set<Node> visited = new HashSet<>();
        // build topology
        {
            buildTopo(node, topology, visited);
        }
        // backward
        for (Node n : topology) {
            if (n instanceof OperationNode opNode) {
                Value v = opNode.getValue();
                Operation op = opNode.getOperation();
                List<Value> inputs = opNode.getInputNodes()
                        .stream()
                        .map(Node::getValue)
                        .toList();
                op.backward(v, inputs);
            }
        }
    }

    private void buildTopo(Node node, Deque<Node> topology, Set<Node> visited) {
        if (visited.contains(node)) {
            return;
        }
        visited.add(node);
        // This ordering guarantees that we don't use premature upstream gradients to compute subsequent gradients
        if (node instanceof OperationNode operationNode) {
            for (Node input : operationNode.getInputNodes()) {
                buildTopo(input, topology, visited);
            }
            topology.addFirst(node); // add node AFTER all its inputs have been added
        }
    }
    ...
}
```

Now we will implement the `backward` method for all of our operations. We will first derive how to compute the local gradinets for each operation and the implement the backward pass according to our findings.

The local gradients for the plus operation are computed as follows:
$$
\frac{\partial}{\partial a}(a+b)=1
$$
$$
\frac{\partial}{\partial b}(a+b)=1
$$
Given that the gradients of a plus operation are always one, applying the chain rule simplifies to simply the upstream gradient to the operation.
The plus operation is thus often interpreted as a "gradient router", as it simply distributes the upstream gradients to all of its inputs.

```java
public class PlusOp {
    ...
    @Override
    public void backward(Value c, Value a, Value b) {
        double upstreamGradient = c.getGrad();
        a.accumulateGrad(upstreamGradient);
        b.accumulateGrad(upstreamGradient);
    }
}
```

The gradients for the multiply operation are computed as follows:
$$
\frac{\partial}{\partial a}(ab)=b
$$
$$
\frac{\partial}{\partial b}(ab)=a
$$
In the case of the multiplication operation, the local gradients are simply the other input value respectively.

```java
public class MultiplyOp {
    ...
    @Override
    public void backward(Value c, Value a, Value b) {
        double upstreamGradient = c.getGrad();
        a.accumulateGrad(upstreamGradient * b.getValue());
        b.accumulateGrad(upstreamGradient * a.getValue());
    }
}
```

The gradients for the power operation are computed as follows:
$$
\frac{\partial}{\partial a}(a^b)=b\cdot a^{b-1}
$$
$$
\frac{\partial}{\partial b}(a^b)=a^b\cdot\ln(a)
$$

```java
public class PowOp {
    ...
    @Override
    public void backward(Value c, Value a, Value b) {
        double upstreamGradient = c.getGrad();
        // Power rule: d/da (a^b) = y * a^(b-1)
        a.accumulateGrad(upstreamGradient * b.getValue() * Math.pow(a.getValue(), b.getValue() - 1));
        // Exponentiation rule: d/db (a^b) = a^b * ln(a)
        b.accumulateGrad(upstreamGradient * Math.pow(a.getValue(), b.getValue()) * Math.log(a.getValue()));
    }
}
```

We will now test the capabilities of our autograd engine with the following example:

```java
public class Main {
    public static void main(String[] args) {
        Value a = new Value(2);
        Value b = new Value(4);

        Value c = a.multiply(b);

        Value d = new Value(2);
        Value e = c.pow(d);

        e.backward();

        System.out.println("a = " + a);
        System.out.println("b = " + b);
        System.out.println("c = " + c);
        System.out.println("d = " + d);
        System.out.println("e = " + e);
    }
}
```

The output of this program is as follows:

```
a = Value{v=2.0, grad=64.0}
b = Value{v=4.0, grad=32.0}
c = Value{v=8.0, grad=16.0}
d = Value{v=2.0, grad=133.0842586675095}
e = Value{v=64.0, grad=1.0}
```

As we can see, the gradients of all the `Value` objects in the graph have been computed correctly.

We can now create a simple abstraction over the `Value` class to implement a linear layer, much like in Sci-Core, except on a scalar level.
    
```java
public class Linear {
    ...
        this.inputsSize = inputSize;
        this.outputsSize = outputSize;
        this.weights = new Value[outputSize][inputSize];
        this.biases = new Value[outputSize];
        Random random = new Random(123);
        double k = 1.0 / Math.sqrt(inputSize);
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = new Value(random.nextDouble() * 2 * k - k);
            }
        }
        for (int i = 0; i < outputSize; i++) {
            biases[i] = new Value(0);
        }
    ...

    public Value[][] forward(Value[][] inputs) {
        // matrix multiplication D=WX
        // D = (batchSize, outputSize)
        // W = (outputSize, inputSize)
        // X = (batchSize, inputSize)
        Value[][] outputs = new Value[inputs.length][outputsSize];
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < outputsSize; j++) {
                Value sum = new Value(0);
                for (int k = 0; k < inputsSize; k++) {
                    sum = sum.plus(weights[j][k].multiply(inputs[i][k]));
                }
                outputs[i][j] = sum.plus(biases[j]);
            }
        }
        return outputs;
    }
    ...
}
```

We will now construct a simple neural network that learns to approximate the function $f(x)=2*x^2 + 0.5$ - only in the value range $[0, 1]$ (this is to avoid saturation of the sigmoid function).

```java

public class SimpleNNTest {
    public static void main(String[] args) {

        class BobNet implements Module {

            private final Sigmoid act = new Sigmoid();
            private final Linear fc1 = new Linear(1, 1);
            private final Linear fc2 = new Linear(1, 1);

            Value[][] forward(Value[][] x) {
                Value[][] h;
                h = fc1.forward(x);
                h = act.forward(h);
                h = fc2.forward(h);
                return h;
            }
            ...
        }

        BobNet bobNet = new BobNet();

        Random random = new Random(123);

        int batchSize = 32;

        // training loop
        for (int step = 0; step < 4_000; step++) {
            Value[][] x = new Value[batchSize][1];
            Value[][] y = new Value[batchSize][1];
            for (int i = 0; i < batchSize; i++) {
                // f(x) = 2 * x^2 + 0.5
                float xVal = random.nextFloat();
                float yVal = 2 * (xVal * xVal) + 0.5f;
                x[i][0] = new Value(xVal);
                y[i][0] = new Value(yVal);
            }
            Value[][] yHat = bobNet.forward(x);
            Value loss = yHat[0][0].minus(y[0][0]).pow(new Value(2));
            loss.backward();

            if (step % 10 == 0) {
                System.out.println("step = " + step + ", loss = " + loss.getValue());
            }

            // sgd
            for (Value parameter : bobNet.getParameters()) {
                parameter.setValue(parameter.getValue() - 0.1 * parameter.getGrad());
                parameter.zeroGrad();
            }
        }

        // test
        Value[][] x = new Value[][]{{new Value(0.13)}};
        Value[][] yHat = bobNet.forward(x);
        System.out.println("yHat: " + yHat[0][0].getValue() + " (expected: 0.5169)");
    }
}
```
The output of this program is as follows:

```
step = 0, loss = 1.6565700714037799
step = 100, loss = 0.09935483659101839
step = 200, loss = 0.022114350721041513
step = 300, loss = 0.020603378421378514
...
step = 3700, loss = 1.5217983452037842E-5
step = 3800, loss = 2.2334898685634553E-4
step = 3900, loss = 5.151804926115063E-5
yHat: 0.5200816868293555 (expected: 0.5169)
```

We see that the loss is decreasing and the network is learning to approximate the function correctly.
Note that this is a very simple example, as scalar-based autograd does not scale well to deep neural networks.

## Tensor-level Autograd
Now we will take a look at the concept of an autograd engine where the atomic unit of differentiation are not individual scalars, but rather tensor-level, higher level operations such as eg. matrix multiplication.

Given a well defined operator $f(P_1, P_2, ..., P_n)$ which computes an output $Z$, each parameter $P_n$ will receive a local gradient tensor $G_n$ of the same shape, where each element $G_{ijk...n}$ is the partial derivative $\frac{\partial L}{\partial P_{ijk...n}}$.
This is a special case of a Jacobian matrix, where the final value $L$ that we are differentiating with respect to is a scalar.
In the general case, the Jacobian matrix is a matrix of partial derivatives of the output $Z$ with respect to each parameter $P$.
The Jacobian matrix is usually defined as the cartesian product of the partial derivatives of a vector valued function $f(p_1, p_2, ... p_n)$ with respect to each scalar valued parameter $p$.
$$
J=
\begin{bmatrix}
\frac{\partial Z_1}{\partial p_1} & \frac{\partial Z_1}{\partial p_2} & \cdots & \frac{\partial Z_1}{\partial p_n} \\
\frac{\partial Z_2}{\partial p_1} & \frac{\partial Z_2}{\partial p_2} & \cdots & \frac{\partial Z_2}{\partial p_n} \\
\cdots & \cdots & \ddots & \cdots \\
\frac{\partial Z_m}{\partial p_1} & \frac{\partial Z_m}{\partial p_2} & \cdots & \frac{\partial Z_m}{\partial p_n}
\end{bmatrix}
$$
Now, this is slightly too specialized and generalized for our purposes at the same time.
We would like a matrix of partial derivatives of a scalar valued function $f(P_1, P_2, ..., P_n)$ with respect to each tensor-valued parameter $P$.
But, when each tensor-valued parameter $P$ is decomposed into its individual elements $P_{ijk...n}$, we can utilize the concept of the Jacobian matrix for our purposes.
When $Z$ is a scalar ($L$), the Jacobian matrix is a vector of partial derivatives of the scalar valued function $f(p_1, p_2, ... p_n)$ with respect to each scalar valued parameter $p$.
$$
J=
\begin{bmatrix}
\frac{\partial L}{\partial p_1}
\frac{\partial L}{\partial p_2}
\cdots
\frac{\partial L}{\partial p_n}
\end{bmatrix}
$$
When we re-compose the elements corresponding to the individual parameters $P$, we arrive at a vector of un-equal sized vectors of partial derivatives, where each element is the flattened version of the derivative of the respective parameter $P$.

$$
J=
\begin{bmatrix}
\begin{bmatrix}
\frac{\partial L}{\partial p^{(1)}_{1}}
\frac{\partial L}{\partial p^{(1)}_{2}}
\cdots
\frac{\partial L}{\partial p^{(1)}_{n}}
\end{bmatrix}
\begin{bmatrix}
\frac{\partial L}{\partial p^{(2)}_{1}}
\frac{\partial L}{\partial p^{(2)}_{2}}
\frac{\partial L}{\partial p^{(2)}_{3}}
\cdots
\frac{\partial L}{\partial p^{(2)}_{n}}
\end{bmatrix}
\cdots
\begin{bmatrix}
\frac{\partial L}{\partial p^{(m)}_{1}}
\frac{\partial L}{\partial p^{(m)}_{2}}
\cdots
\frac{\partial L}{\partial p^{(m)}_{n}}
\end{bmatrix}
\end{bmatrix}
$$
The elements $J_i$ are the global gradients of the respective parameter $P_i$, which we can interpret as tensors of the same shape as the respective parameter $P$.
As the chain rule also applies to Jacobians, we multiply the local gradient with the upstream gradient to compute the global gradient.
Note that $upstreamGradient * localGradient$ must result in a shape equal to the shape of $P_i$. This will not happen when some dimension $d$ is broadcast in the operation $f$ such that $|P_d| = 1$ and $|Z_d|$ > 1. Because a broadcast is the virtual repetition of a scalar value of a parameter $P_i$ in the dimension $d$ to match the dimension size of another operation $P_j$ at the respective dimension, all $d > 1$ gradient values that would be associated with the individual repition of the same scalar value must be sumed up to collect all gradient contributions. Inversely, when a parameter $P$ is summed up along a dimension $n$, such that $|P_d| = 1$ (or the dimension not kept) and $|Z_d|$ > 1, the gradient of the single resulting scalar must be repeated in the dimension $d$. We can therefore conclude that summation and broadcast operations are their own inverse operations during backpropagation.

Similar to the scalar-based autograd, the tensor-based autograd engine in Sci-Core builds a graph according to the method calls performed on the tensor objects using a `GraphRecorder` object.

```java
public class GraphRecorder {
    ...
    @Override
    @NotNull
    public ITensor matmul(@NotNull ITensor other) {
        ISciCoreBackend backend = getSciCoreBackend();
        IGraphRecorder operationRecorder = backend.getOperationRecorder();
        return operationRecorder.recordOperation(OperationType.MATMUL, backend, this, other);
    }
    ...
}
```

After calling the operation methods on the tensor objects, the recorded graph can retrieved. Gradients can be requested to 
be computed by the autograd engine.
After starting the backpropagation, the gradients can be retrieved for all tensors which gradients were requested for.

```java
ITensor a = sciCore.matrix(new float[][]{{1, 2, 3, 4}});
ITensor b = sciCore.matrix(new float[][]{{5, 6}, {8, 9}, {11, 13}, {15, 17}});
ITensor c = a.matmul(b);

ITensor d = sciCore.matrix(new float[][]{{1}, {3}});
ITensor e = c.matmul(d);

IGraph graph = sciCore.getExecutionGraphUpTo(e);
graph.requestGradientsFor(a, b);
graph.backward();
```

The graph can be visualized using the `GraphVisualizer` class:

```java
GraphVisualizer.saveGraph(DAGGraphRenderPlanFactory.makeRenderPlan(graph), "graph.png");
```

![opgraph_1](./figures/opgraph_1.png)

Only branches of the graph that lead to explictly requested gradients will be computed to avoid unnecessary computation.
Only the gradients of the requested tensors persist after the backward pass completes.

```java
public class Graph {
    ...
    @Override
    public void requestGradientsFor(@NotNull List<ITensor> parameters) {
        List<ITensorNodeWithGradient> parameterNodes = new ArrayList<>(parameters.size());
        for (ITensor parameter : parameters) {
            Optional<IGraphNode> nodeOpt = getNodeForTensor(parameter);
            if (nodeOpt.isEmpty()) {
                throw new IllegalArgumentException("Parameter not found in graph");
            }
            IGraphNode parameterNode = nodeOpt.get();
            if (!(parameterNode instanceof ITensorNodeWithGradient parameterNodeWithGradient)) {
                throw new IllegalArgumentException("Parameter is not a differentiable tensor");
            }
            parameterNodes.add(parameterNodeWithGradient);
        }
        for (ITensorNodeWithGradient parameterNode : parameterNodes) {
            parameterNode.requestGradients();
            parameterNode.setRequireGradients();
            Set<IGraphNode> downstreamNodes = parameterNode.getDownstreamNodes();
            Set<IGraphNode> visitedNodes = new HashSet<>();
            Queue<IGraphNode> queue = new ArrayDeque<>(downstreamNodes);
            while (!queue.isEmpty()) {
                IGraphNode node = queue.poll();
                if (visitedNodes.contains(node)) {
                    continue;
                }
                visitedNodes.add(node);
                if (node instanceof ITensorNodeWithGradient tensorNodeWithGradient) {
                    tensorNodeWithGradient.setRequireGradients();
                }
                queue.addAll(node.getDownstreamNodes());
            }
        }
    }
    ...
}
```
Nodes can either request gradients, or require gradients. A node that requests gradients will have its gradients computed and not discarded. A node that requires gradients will have its gradients computed and discarded, as these gradients are only needed to compute the gradients of upstream nodes, one of which will request gradients.

These flags are respected in the backpropagate method to only traverse branches of the graph where nodes depend on nodes that request gradients.
This also handles the case of deleting gradients in `clearUnusedGradients()`.
```java
public class Graph {
    ...
    @Override
    public void backward() {
        // initialize gradient to 1 because derivative of x in respect to itself is one. Duh.
        if (outputNode instanceof ITensorNodeWithGradient nodeWithGradient) {
            ITensor tensor = nodeWithGradient.getValue();
            if (!tensor.isScalar()) {
                throw new IllegalStateException("Cannot compute gradient of non-scalar tensor");
            }

            ITensor gradient = backend.createTensor(tensor.getDataType(), tensor.getShape());
            gradient.fill(1);
            nodeWithGradient.accumulateGradient(gradient); // dL/dL = 1

            // apply chain rule
            backPropagate(nodeWithGradient);

            // collect results
            Set<ITensor> gradients = collectGradientResults();

            // clear gradients
            clearUnusedGradients(gradients);
        } else {
            throw new IllegalStateException("Output node of graph must be differentiable!");
        }
    }
    ...
    private void backPropagate(@NotNull ITensorNodeWithGradient node) {
        Deque<IGraphNode> topology = new LinkedList<>();
        Set<IGraphNode> visited = new HashSet<>();
        // build topology
        {
            buildTopo(node, topology, visited);
        }
        // back propagate
        for (IGraphNode currentNode : topology) {
            if (currentNode instanceof ITensorNodeWithGradient currentNodeWithGradient) {
                // only compute gradient for nodes for which it is required
                if (currentNodeWithGradient.requiresGradients()) {
                    if (currentNode instanceof IDifferentiableNode differentiableNode) {
                        differentiableNode.computeGradients();
                    }
                }
            }
        }
    }
    ...
    private void buildTopo(IGraphNode node, Deque<IGraphNode> topology, Set<IGraphNode> visited) {
        if (visited.contains(node)) {
            return;
        }
        visited.add(node);
        // This ordering guarantees that we don't use premature upstream gradients to compute subsequent gradients
        if (node instanceof OperationGraphNode operationNode) {
            for (IGraphNode input : operationNode.getInputs()) {
                buildTopo(input, topology, visited);
            }
            topology.addFirst(node); // add node AFTER all its inputs have been added
        }
    }
    ...
}
```

As the individual operations operate on a tensor-basis, their backward passes are also more complex than those of their scalar counterparts. Mechanisms like broadcasting must be accounted for in the fashion described above, and more complex operation require a rigerous mathematical deriviation to simplify gradient computation.

The following snippet shows the backward pass of the multiplication operation in Sci-Core:

```java
public class GenCPUMultiplyOp implements IDifferentiableBinaryOperation {
    ...
    @Override
    public void computeGradients(Graph.IOperationContext ctx, ITensor upstreamGradient,
         IGraph.ITensorNodeWithGradient a, IGraph.ITensorNodeWithGradient b) {
        if (a.requiresGradients()) {
            try (ITensor gradients = upstreamGradient.multiply(b.getValue())) {
                ITensor finalGradients = GradientUtil.sumGradientsOnBroadcastDims(gradients, a.getValue().getShape());
                a.accumulateGradient(finalGradients);
            }
        }
        if (b.requiresGradients()) {
            try (ITensor gradients = upstreamGradient.multiply(a.getValue())) {
                ITensor finalGradients = GradientUtil.sumGradientsOnBroadcastDims(gradients, b.getValue().getShape());
                b.accumulateGradient(finalGradients);
            }
        }
    } 
}
```

When compared with the scalar counterpart, the expression $globalGradient = upstreamGradient * localGradient$ is still intact. However, we need to handle broadcasting in addition to that. The `GradientUtil.sumGradientsOnBroadcastDims` method takes care of that. It sums the gradients along the dimensions where the parameter tensor was broadcasted in the forward pass. We deduce these dimensions based on the shape of the parameter tensor and the shape of the temporary gradient tensor.

```java
public class GradientUtil {

    public static ITensor sumGradientsOnBroadcastDims(ITensor tmpGradients, long[] shapeOfParameter) {
        long[] gradientShape = tmpGradients.getShape();
        for (int i = 0; i < gradientShape.length; i++) {
            if (shapeOfParameter.length - i - 1 >= 0) {
                if (gradientShape[gradientShape.length - i - 1] != shapeOfParameter[shapeOfParameter.length - i - 1]) {
                    tmpGradients = tmpGradients.reduceSum(gradientShape.length - i - 1, true);
                }
            } else {
                tmpGradients = tmpGradients.reduceSum(0, false);
            }
        }
        return tmpGradients;
    }
}
```

The method is used in the backward pass to account for broadcasting in the forward pass.
The function takes in `tmpGradients` $\frac{\partial L}{\partial Z}$ and the shape of the parameter $P$ that $Z=f(P, ...)$ depends on and returns $\frac{\partial L}{\partial P}$.

Not all operations are tensor-level backward passes are simple ports of their scalar counterparts. The backward pass of the `matmul` operation is a good example of this.


### Tensor level matrix multiplication differentiation

As a backward pass generally needs to be performant, it is very fortunate that a the backwards pass of a matrix multiplication can itself be expressed as a matrix multiplication.

To prove this, we will resort to interpreting gradient computation as computing the Jacobian for an operation $f$.

Note our specialization of the Jacobian matrix:
$$
J=
\begin{bmatrix}
\begin{bmatrix}
\frac{\partial L}{\partial p^{(1)}_{1}}
\frac{\partial L}{\partial p^{(1)}_{2}}
\cdots
\frac{\partial L}{\partial p^{(1)}_{n}}
\end{bmatrix}
\begin{bmatrix}
\frac{\partial L}{\partial p^{(2)}_{1}}
\frac{\partial L}{\partial p^{(2)}_{2}}
\frac{\partial L}{\partial p^{(2)}_{3}}
\cdots
\frac{\partial L}{\partial p^{(2)}_{n}}
\end{bmatrix}
\cdots
\begin{bmatrix}
\frac{\partial L}{\partial p^{(3)}_{1}}
\frac{\partial L}{\partial p^{(3)}_{2}}
\cdots
\frac{\partial L}{\partial p^{(3)}_{n}}
\end{bmatrix}
\end{bmatrix}
$$

When we are differentiating with respect to only one parameter, we can simplify the Jacobian matrix to a vector:

$$
J=
\begin{bmatrix}
\begin{bmatrix}
\frac{\partial L}{\partial p^{(1)}_{1}}
\frac{\partial L}{\partial p^{(1)}_{2}}
\cdots
\frac{\partial L}{\partial p^{(1)}_{n}}
\end{bmatrix}
\end{bmatrix}
=
\begin{bmatrix}
\frac{\partial L}{\partial p^{(1)}_{1}}
\frac{\partial L}{\partial p^{(1)}_{2}}
\cdots
\frac{\partial L}{\partial p^{(1)}_{n}}
\end{bmatrix}
$$

When we remember that we defined $J_i$ as the flat version of the gradient tensor $G_i$, and that $G_i$ is the gradient tensor for the parameter $P_i$, and that $P_i$ and $G_i$ have the same shape due to $L$ being a scalar, and that all $P_i$ of $matmul(P_1, P_2)$ are matrices, we can re-arange our gradient tensor $G$ to be a matrix for aesthetic reasons.

Given $Z = W \cdot X$ where $W$ is a matrix sized $a \times b$ and $X$ is a matrix sized $b \times c$,
and $L = f(Z)$, we can compute the derivative of the scalar $L$ with respect to $W$ or $X$.

We must now chose which parameter we are differentiating with respect to. Let's first choose $W$.

We can now define the gradient tensor $G_w$ as follows:

$$
G_w=\frac{\partial L}{\partial W}
=\left[
\begin{array}{ccc}
   \frac{\partial L}{\partial W_{11}} & \cdots & \frac{\partial L}{\partial W_{1b}} \\
   \vdots & \ddots & \vdots \\
   \frac{\partial L}{\partial W_{a1}} & \cdots & \frac{\partial L}{\partial W_{ab}}
\end{array}
\right]
$$

Let's first define the scalar case what $Z_{ij}$ is: 
$$
Z_{ij} = \sum_{k=1}^{b} W_{ik} \cdot X_{kj}
$$

Note that $Z$ is a matrix of size $a \times c$.
As we want to compute the derivative of $L$ with respect to $W$ and therefore all of its element simultaneously, we must account for all gradient contributions of all elements of $Z$ with respect to all elements of $W$.

To properly define what this means, we could first approach this problem with simple single-variable calculus.
When given $Z = W \cdot X$, we can create all possible derivatives for all $Z_{kl}$ with respect to all $W_{ij}$.
Given that $i \in [1, a]$ and $j \in [1, b]$ and $k \in [1, a]$ and $l \in [1, c]$, we would arive at $a \times b \times a \times c$ derivatives, which obviously differs from the $a \times b$ elements of $W$, which we expect given our specialization of the Jacobian matrix, where $L$ is a scalar. We thus sum over the remaining dimensions contributing to a respective entry $W_{ij}$.
We thus arrive at the following equation:

$$
\frac{\partial L}{\partial W_{ij}} = \sum_{k=1}^{a}\sum_{l=1}^{c} \frac{\partial L}{\partial Z_{kl}} \cdot \frac{\partial Z_{kl}}{\partial W_{ij}}
$$

When $k \neq i$, then $\frac{\partial Z_{kl}}{\partial W_{ij}} = 0$. This means that we can simplify the above equation to:

$$
\frac{\partial L}{\partial W_{ij}} = \sum_{l=1}^{c} \frac{\partial L}{\partial Z_{il}} \cdot \frac{\partial Z_{il}}{\partial W_{ij}}
$$

We exapand $\frac{\partial Z_{il}}{\partial W_{ij}}$:

$$
\frac{\partial Z_{il}}{\partial W_{ij}} = \frac{\partial}{\partial W_{ij}} \left( \sum_{q=1}^{b} W_{iq} \cdot X_{  ql} \right)
$$

We now see that $\frac{\partial Z_{il}}{\partial W_{ij}} = 0$ as well, if $q \neq j$.


These conditions are quite intuitive, as the derivative will only be $\neq 0$ if the weight we are differentiating with respect to is used in the multiplication term.


We can now simplify the above equation to:

$$
\frac{\partial Z_{il}}{\partial W_{ij}} = \frac{\partial}{\partial W_{ij}} \left( W_{ij} \cdot X_{jl} \right)
=
X_{jl}
$$

We can now substitute this into the above equation:

$$
\frac{\partial L}{\partial W_{ij}} = \sum_{l=1}^{c} \frac{\partial L}{\partial Z_{il}} \cdot X_{jl}
$$

Given that each entry of the resulting matrix computes as the sum of element-whise multiplication of scalars that stem from other matrices via indexing, we can rewrite this operation as the matrix multiplication of said matrices. Note however, that $X$ is of size $b \times c$ and that we index into via $X_{jl}$, where $j \in [1, b]$ and $l \in [1, c]$.
Now however that the sum in the above equation $l$ does not index into the leading dimension of the second matrix, but into the trailing dimension. We must thus transpose $X$ to be of size $c \times b$:

$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial Z} \cdot X^T
$$

We will now repeat the same process for the derivative of $L$ with respect to $X$. Explanation will be omitted for similar steps as above.

$$
G_x=\frac{\partial L}{\partial X}
=\left[
\begin{array}{ccc}
   \frac{\partial L}{\partial X_{11}} & \cdots & \frac{\partial L}{\partial X_{1c}} \\
   \vdots & \ddots & \vdots \\
   \frac{\partial L}{\partial X_{b1}} & \cdots & \frac{\partial L}{\partial X_{bc}}
\end{array}
\right]
$$

$$
\frac{\partial L}{\partial X_{ij}} = \sum_{k=1}^{a}\sum_{l=1}^{c} \frac{\partial L}{\partial Z_{kl}} \cdot \frac{\partial Z_{kl}}{\partial X_{ij}}
$$

If $l \neq j$, then $\frac{\partial Z_{kl}}{\partial X_{ij}} = 0$. This means that we can simplify the above equation to:

$$
\frac{\partial L}{\partial X_{ij}} = \sum_{k=1}^{a} \frac{\partial L}{\partial Z_{ki}} \cdot \frac{\partial Z_{ki}}{\partial X_{ij}}
$$

$$
\frac{\partial Z_{ki}}{\partial X_{ij}} = \frac{\partial}{\partial X_{ij}} \left( \sum_{q=1}^{b} W_{kq} \cdot X_{qj} \right)
$$

If $q \neq i$, then $\frac{\partial Z_{ki}}{\partial X_{ij}} = 0$. This means that we can simplify the above equation to:

$$\frac{\partial Z_{ki}}{\partial X_{ij}} = \frac{\partial}{\partial X_{ij}} \left( W_{ki} \cdot X_{ij} \right)=W_{ki}$$

$$
\frac{\partial L}{\partial X_{ij}} = \sum_{k=1}^{a} \frac{\partial L}{\partial Z_{ki}} \cdot W_{ki}
$$


$$
\frac{\partial L}{\partial X} = \left(\frac{\partial L}{\partial Z}\right)^T \cdot W
$$

To summarize, we derived that
$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial Z} \cdot X^T
$$
and
$$
\frac{\partial L}{\partial X} = \left(\frac{\partial L}{\partial Z}\right)^T \cdot W
$$

We will now use this insight to implement an efficient backward pass for the matrix multiplication operation.

```java
public class GenCPUMatmulOp implements IDifferentiableOperation {
    ...
    @Override
    public void computeGradients(Graph.IOperationContext ctx,
        ITensor upstreamGradient,
        IGraph.ITensorNodeWithGradient a, IGraph ITensorNodeWithGradient b) {
        OptionBundle options = ctx.getOptionBundle();
        boolean transposeA = options.getOrDefault("transposeA", false);
        boolean transposeB = options.getOrDefault("transposeB", false);
        
        ITensor aValue = a.getValue();
        ITensor bValue = b.getValue();

```
Note that we want to support virtual transpose operations, that do not actually transpose the underlying tensor, but rather cleverly express the backward pass of virtually transposed matrix multiplication as virtually transpose matrix multplications themselves.
Getting this right envolves additional derivations.
We will now refer to the upstream gradient $\frac{\partial L}{\partial Z}$ as $G$, where $Z = W \cdot X$.

First we start with the base case:
$$
\frac{\partial L}{\partial W} = G \cdot X^T
$$

$
if \space \text{transposeA} = False: \newline
\qquad if \space \text{transposeB} = True: \newline
$
The base case only applies when the transpose was actually applied to the op input before the matrix multiplication.
During the forward pass a "virtual transpose" occurred, but this is not reflected in the graph. Thus, we need to transpose $X$ again.

$
\qquad \qquad \frac{\partial L}{\partial W} = G \cdot (X^T)^T = G \cdot X \newline
\qquad else: \newline
$
No virtual transpose occurrs here, because $X$ here is what was actually used in the forward pass.

$
\qquad \qquad \frac{\partial L}{\partial W} = G \cdot X^T
\newline
else \space if \space \text{transposeA} = True: \newline
$
Normally, if $X$ were a transpose operation node, this would compute the upstream gradients, which would transpose it again as part of its gradient computation.
However, since we are merging a defacto transpose operation into the matmul operation, we would need to transpose
these gradients after $\frac{\partial L}{\partial W}$ is computed. We also exploit following identity:
$
B^T \cdot A^T = (A \cdot B)^T
$.
This allows us to represent $(A \cdot B)^T$ in terms of things we can cheaply compute, namely $A^T$ and $B^T$ as part of a virtually transposed matrix multiplication operation.

$
\qquad if \space \text{transposeB} = True: \newline
$
Here a virtual transpose occurred, so we need to transpose $X$ again.
Note that in addition to that we also need to transpose again because of the would-be transpose operation node, which in this configuration does not exist in the graph.

$
\qquad \qquad \frac{\partial L}{\partial W} = (G \cdot (X^T)^T)^T =
(G \cdot X)^T
= X^T \cdot G^T \newline
\qquad else: \newline
$
Here no virtual transpose occured, but we still need to account for the would-be transpose operation node and transpose the result an expression.

$
\qquad \qquad \frac{\partial L}{\partial W} = (G \cdot X^T)^T = X \cdot G^T 
$

These deriviations can be summarized as follows:

$
\frac{\partial L}{\partial W} = \begin{cases}
G \cdot X & \text{if } \text{transposeA} = False \text{ and } \text{transposeB} = True \newline
G \cdot X^T & \text{if } \text{transposeA} = False \text{ and } \text{transposeB} = False \newline
X^T \cdot G^T & \text{if } \text{transposeA} = True \text{ and } \text{transposeB} = True \newline
X \cdot G^T & \text{if } \text{transposeA} = True \text{ and } \text{transposeB} = False \newline
\end{cases}
$

We note that when we group the cases by the value of $\text{transposeA}$, we can see that computation of the gradients involve the same operand matrices,
but with different transpositions. This serves our purpose well, as we can now derive the following:

Given a matrix multiplication $C = A \cdot B$, where a tuple $(X, Y)$ of boolean states represents the transposition of $A$ and $B$ respectively, 
and $\frac{\partial L}{\partial W}$ can be computed as some $A \cdot B$ with transposition states $(X, Y)$, we can derive the following:

$
\begin{cases}
G \cdot X \quad with \quad (False, \neg transposeB) \space & if \space \text{transposeA} = False \newline
X \cdot G \quad with \quad (transposeB, \space True) \space &if \space \text{transposeA} = True \newline
\end{cases}
$

We can now turn this into code:

```java
        if (a.requiresGradients()) {
            ITensor dLdW;
            if (!transposeA) {
                dLdW = upstreamGradient.matmul(bValue, false, !transposeB);
            } else {
                dLdW = bValue.matmul(upstreamGradient, transposeB, true);
            }
```

In the following lines, we handle the case where at least one of the operands is a 3D tensor, where the first dimension is interpreted as a batch dimension,
which allows for efficient computation of multiple matrix multiplication operations of the same shape in parallel. Note that this produces an output tensor, where the first dimension is the batch dimension. We need to sum over this dimension to accumulate gradient contributions from all batch elements such that the gradient tensor has the same shape as the operand tensor.

```java
            if (aValue.getShape().length == 3 || bValue.getShape().length == 3) {
                if (aValue.getShape().length == 3 && aValue.getShape()[0] == 1) {
                    dLdW = dLdW.reduceSum(0, true);
                } else if (aValue.getShape().length == 2) {
                    dLdW = dLdW.reduceSum(0, false);
                }
            }
            a.accumulateGradient(dLdW);
        }
```

When differentiating with respect to the second operand, we can use the same logic as for the first operand, but starting from a different base case,
which is $\frac{\partial L}{\partial X} = W^T \cdot G$.
Explanation of the derivation is omitted here, as it is similar to the derivation for the first operand.

$
if \space \text{transposeB} = False: \newline
\qquad if \space \text{transposeA} = True: \newline
\qquad \qquad \frac{\partial L}{\partial X} = (W^T)^T \cdot G = W \cdot G \newline
\qquad else: \newline
\qquad \qquad \frac{\partial L}{\partial X} = W^T \cdot G \newline
else: \newline
\qquad if \space \text{transposeA} = True: \newline
\qquad \qquad \frac{\partial L}{\partial X} = ((W^T)^T \cdot G)^T = (W \cdot G)^T = G^T \cdot W^T \newline
\qquad else: \newline
\qquad \qquad \frac{\partial L}{\partial X} = (W^T \cdot G)^T = (W^T \cdot G)^T = G^T \cdot W \newline
$

$
\frac{\partial L}{\partial X} = \begin{cases}
W \cdot G & \text{if } \text{transposeA} = True \text{ and } \text{transposeB} = False \newline
W^T \cdot G & \text{if } \text{transposeA} = False \text{ and } \text{transposeB} = False \newline
G^T \cdot W^T & \text{if } \text{transposeA} = True \text{ and } \text{transposeB} = True \newline
G^T \cdot W & \text{if } \text{transposeA} = False \text{ and } \text{transposeB} = True \newline
\end{cases}
$

$
\begin{cases}
W \cdot G \quad with \quad (\neg transposeA, \space False) \space & if \space \text{transposeB} = False \newline
G \cdot W \quad with \quad (True, \space transposeB) \space & if \space \text{transposeB} = True \newline
\end{cases}
$

```java
        if (b.requiresGradients()) {
            ITensor dLdX;
            if (!transposeB) {
                dLdX = aValue.matmul(upstreamGradient, !transposeA, false);
            } else {
                dLdX = upstreamGradient.matmul(aValue, true, transposeA);
            }
            if (aValue.getShape().length == 3 || bValue.getShape().length == 3) {
                if (bValue.getShape().length == 3 && bValue.getShape()[0] == 1) {
                    dLdX = dLdX.reduceSum(0, true);
                } else if (bValue.getShape().length == 2) {
                    dLdX = dLdX.reduceSum(0, false);
                }
            }
            b.accumulateGradient(dLdX);
        }
    }
}
```
### Optimization
After computing gradients, we can implement first-order optimization algorithms, such as stochastic gradient descent to update the parameters of the model in direction of the negative gradient.
We can implement a simple `SGD´ optimizer as follows:

```java
public class Sgd implements IOptimizer {

    ...

    @Override
    public void step(@NotNull ITensor loss) {
        try (IGraph graph = sciCore.getBackpropagationGraphUpTo(loss, parameters)) {
            sciCore.getBackend().getOperationRecorder().recordWithScope(() -> {
                graph.backward();
                for (ITensor parameter : parameters) {
                    try (ITensor gradient = graph.getGradient(parameter)
                            .orElseThrow(() -> new IllegalStateException("No gradient for parameter"))) {
                        float learningRate;
                        if (adaptiveLearningRate) {
                            learningRate = (float) (initialLearningRate * Math.pow(learningRateDecayFactor, nSteps));
                        } else {
                            learningRate = this.initialLearningRate;
                        }
                        try (ITensor scaledGradient = gradient.multiply(learningRate)) {
                            parameter.subtract(scaledGradient);
                        }
                    }
                }
                return null;
            });
            nSteps++;
        }
    }
}
```

### Mnist Training Example
We can now take these building blocks to create a simple multi-layer neural network, which we can train to recognize handwritten digits from the MNIST dataset.

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

...

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
                .use { loss ->
                    optimizer.step(loss)
                    loss.elementAsDouble()
                }
        }
    }
}
```
Training the model for 60,000 steps, we can achieve an accuracy of 95.5% on the test set.

```
MNIST already downloaded
[16:10:35] [main/DEBUG]: Operation MULTIPLY found in backend GenCPUBackend
[16:10:35] [main/DEBUG]: Operation MINUS found in backend GenCPUBackend
Start training...
[16:10:35] [main/DEBUG]: Operation MATMUL found in backend GenCPUBackend
[16:10:35] [main/DEBUG]: Operation PLUS found in backend GenCPUBackend
[16:10:35] [main/DEBUG]: Operation RELU found in backend GenCPUBackend
[16:10:35] [main/DEBUG]: Operation EXP found in backend GenCPUBackend
[16:10:35] [main/DEBUG]: Operation REDUCE_SUM found in backend GenCPUBackend
[16:10:35] [main/DEBUG]: Operation DIVIDE found in backend GenCPUBackend
[16:10:35] [main/DEBUG]: Operation POW found in backend GenCPUBackend
[16:10:35] [main/DEBUG]: Operation PLUS_INPLACE found in backend GenCPUBackend
[16:10:35] [main/DEBUG]: Operation MINUS_INPLACE found in backend GenCPUBackend
Training 100% |█████████████████| 60000/60000 (0:00:21 / 0:00:00) loss: 0.00947
Training time: 21.992s
Final loss value: 0.009466836228966713
Examples per second: 87304.47435431066
Start testing...
(0:00:00 / ?) [16:10:57] [main/DEBUG]: Operation ARGMAX found in backend GenCPUBackend
[16:10:57] [main/DEBUG]: Operation COMPARE_ELEMENTS found in backend JvmBackend
[16:10:57] [main/DEBUG]: Operation CAST found in backend GenCPUBackend
[16:10:57] [main/DEBUG]: Operation RESHAPE found in backend GenCPUBackend
Testing 100% |██████████████| 20000/20000 (0:00:02 / 0:00:00) accuracy: 0.95475
Final Accuracy: 0.9547
```

![mnist_inference_screenshot](figures/mnist_inference_screenshot.png)

### Performance considersations

Note that the training only takes 21 seconds on my M1 MacBook Pro using the CPU backend with ARM Neon SIMD and Apple Accelerate optimizations and
only 17 seconds on my AMD Ryzen 5800X desktop using the CPU backend with AVX2 SIMD and Intel MKL optimizations.
If no platform specific matrix multiplication optimizations are available, we fall back on my simple BLAS implementation called "TinyBLAS"
Note that TinyBLAS implements matrix multiplication in a rather naive way, with only low hanging fruit optimizations, which we will discuss later.
TinyBLAS also handles SIMD optimizations for less complicated operators, like multiplication and addition etc. while respecting the strides of a tensor
data structure as well as handling broadcasting in an efficient manner.


The following snippet shows how matrix multiplication is delegated to Apple Accelerate/Intel MKL, which both implement a common BLAS interface:
```cpp
#if defined(__APPLE__)
// if macOS, use Accelerate framework
#include <Accelerate/Accelerate.h>
#elif defined(USE_MKL)
// if MKL is available, use it
#include <mkl_cblas.h>
#else
// Fall back on TinyBLAS
#define USE_TINYBLAS
#endif

JNIEXPORT void JNICALL
Java_me_mikex86_scicore_backend_impl_genericcpu_jni_MatmulJNI_nmatmul(
    JNIEnv *jniEnv, jclass, jint transa, jint transb,
    jint m, jint n, jint k,
    jlong alphaPtr,
    jlong aPtr,
    jint aType,
    jint lda,
    jlong betaPtr, jlong bPtr,
    jint bType,
    jint ldb,
    jlong cPtr,
    jint cType,
    jint ldc) {
    ...
#ifdef USE_TINYBLAS
        tblas_sgemm(TblasRowMajor,
                    transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                    transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                    m, n, k,
                    *(float *) alphaPtr, (float *) aPtr, lda,
                    (float *) bPtr, ldb, *(float *) betaPtr,
                    (float *) cPtr, ldc);
#else
        cblas_sgemm(CblasRowMajor,
                    transa == OP_TRANSPOSE ? CblasTrans : CblasNoTrans,
                    transb == OP_TRANSPOSE ? CblasTrans : CblasNoTrans,
                    m, n, k,
                    *(float *) alphaPtr, (float *) aPtr, lda,
                    (float *) bPtr, ldb, *(float *) betaPtr,
                    (float *) cPtr, ldc);
#endif
    ...
}
```

#### Optimizing Matrix Multiplication for modern CPUs
TinyBLAS strikes a compromise between performance and portability when implementing matrix multiplication as to not use architecture-specific instructions, implementing the algorithm in common C++ and hoping for
compiler optimization hits. This is because the matrix multiplication algorithm of TinyBLAS is designed to be a fallback when no platform specific matrix multiplication optimizations are available.
However, we can make basic assumtions that hold true for most modern CPUs and use them to optimize the matrix multiplication algorithm.

TinyBLAS' matrix multiplication algorithm is in part based on the techniques presented in https://siboehm.com/articles/22/Fast-MMM-on-CPU

```cpp
template<typename A, typename B, typename C>
FORCE_INLINE void tblas_gemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                             int m, int n, int k,
                             A alpha, const A *a, int lda,
                             const B *b, int ldb, B beta,
                             C *c, int ldc) {
    if (beta != 1.0f) {
        if (beta == 0.0f) {
            memset(c, 0, m * n * sizeof(float));
        } else {
            for (int i = 0; i < m * n; i++) {
                c[i] *= beta;
            }
        }
    }
    switch (order) {
        case TblasRowMajor: {
            for (int row = 0; row < m; row++) {
                for (int inner = 0; inner < k; inner++) {
                    for (int col = 0; col < n; col++) {
                        int aIdx = transa == TblasNoTrans ? row * lda + inner : inner * lda + row;
                        int bIdx = transb == TblasNoTrans ? inner * ldb + col : col * ldb + inner;
                        c[row * ldc + col] += alpha * a[aIdx] * beta * b[bIdx];
                    }
                }
            }
            break;
        }
        // Column major order ommitted for brevity
    }
}
```

The above algorithm differs from naive matrix multiplication in subtle, but important ways.
For example, a naive matrix multiplication algorithm that closely represents peoples intuitions about how
to perform matrix multiplication by hand could look like this:
```cpp
for (int row = 0; row < rows; row++) {
    for (int col = 0; col < columns; col++) {
      float acc = 0.0;
      for (int inner = 0; inner < inners; inner++) {
        acc += left[row * columns + inner] * right[inner * columns + col];
      }
      result[row * columns + col] = acc;
    }
}
```
The fatal flaw of the above algorithm is that it is not cache friendly at all.
On modern CPUs, every memory access loads at least 64 bytes of data into the cache, which is called a cache line.
If only a small set of those 64 bytes is actually used before the cache line is evicted, we have to take an expensive trip to system memory.
Note that in modern CPUs the speed of accessing system memory is orders of magnitude slower than accessing any cache tier.
As CPU speeds increase, memory speeds have not kept up, requiring CPU manufacturers to resort to ever larger cache sizes.
Most of the execution time of modern application comes from "pointer chasing",
a specific pattern of incredibly random memory access through a fragmented heap that is impossible be speculatively predicted by the CPU to a degree where it would remove system memory as the bottleneck -
something that most modern languages like Java, Python, JavaScript, etc. are heavily prone to.
It is thus important to keep memory access patterns as regular as possible, ideally completely sequential.

When looking at the inner loop of the above algorithm, we can see that while the inner loop strides over `left` sequentially,
it strides over `right` in constant offsets of `columns * sizeof(float)` bytes.
This wastes fetched cache lines to a catastrophic degree.

To show how catastrophic in fact this is, we will look at the following benchmark:
```java
public class FloatMatmulPerformanceTest {

    static {
        LibraryLoader.loadLibrary("scicore_genericcpu");
    }

    public static void main(String[] args) {
        long alphaPtr = JEmalloc.nje_malloc(4);
        MemoryUtil.memPutFloat(alphaPtr, 1.0f);
        int size = 1024;
        for (int i = 0; i < 1000; i++) {
            FloatBuffer a = JEmalloc.je_malloc(size * size * 4).asFloatBuffer();
            FloatBuffer b = JEmalloc.je_malloc(size * size * 4).asFloatBuffer();
            FloatBuffer c = JEmalloc.je_malloc(size * size * 4).asFloatBuffer();
            // fill random
            for (int j = 0; j < a.remaining(); j++) {
                a.put(j, (float) Math.random());
                b.put(j, (float) Math.random());
            }
            long start = System.nanoTime();
            matmul(OP_NONE, OP_NONE, size, size, size,
                    alphaPtr, MemoryUtil.memAddress(a), MATMUL_DATA_TYPE_FLOAT32, size,
                    alphaPtr, MemoryUtil.memAddress(b), MATMUL_DATA_TYPE_FLOAT32, size,
                    MemoryUtil.memAddress(c), MATMUL_DATA_TYPE_FLOAT32, size);
            long end = System.nanoTime();
            long nFlops = 2L * size * size * size;
            double tflops = (nFlops / ((end - start) / 1e9)) / 1e12;
            System.out.println("matmul took " + (end - start) / 1e6 + " ms, " + tflops + " TFLOPS");
            JEmalloc.je_free(a);
            JEmalloc.je_free(b);
            JEmalloc.je_free(c);
        }
        JEmalloc.nje_free(alphaPtr);
    }
}
```

When running this benchmark with the optimized TinyBLAS matrix multiplication implementation on my M1 MacBook Pro (which has on chip memory,
where the impact of this would be less severe than on any other computer currently on the market),
we get the following results:

```
matmul took 85.514667 ms, 0.02511245992456475 TFLOPS
matmul took 85.637084 ms, 0.025076562018389138 TFLOPS
matmul took 83.41375 ms, 0.025744959889706435 TFLOPS
matmul took 84.584334 ms, 0.02538866887572822 TFLOPS
```

When switching to the naive matrix multiplication implementation, we get the following results:

```
matmul took 1554.149083 ms, 0.0013817745488448742 TFLOPS
matmul took 1556.108334 ms, 0.001380034796471953 TFLOPS
matmul took 1551.034667 ms, 0.0013845490972510934 TFLOPS
matmul took 1555.592959 ms, 0.0013804920082567693 TFLOPS
```

We note that this was a simple loop-reordering of the inner loop that leads
to a 20x speedup in the matrix multiplication benchmark.

We could further optimize the algorithm by using a technique called tiling, where we split the matrix into smaller blocks, preventing values from being prematurely evicted from the cache, but this is beyond the scope of a rather general matrix multiplication algorithm,
which only serves as a fallback for more specialized vendor-specific implementations.
Vendor-specific implementations such as in the case of the Intel MKL library will not shy away from implementing many different algorithms for different matrix sizes and shapes and processor generations and choose between them dynamically at runtime based on cpuid checks etc.

# Language Modelling
What we have implemented so far is a tensor processing library that allows us to optimize scalar-valued functions with tensor-valued parameters, such as the loss function of a neural network. We can now use this approach to optimize a model with predictive power over a sequence of tokens or characters. Such a model is called a neural language model.
But before we dive into what a neural language model is, we will first define language modelling as a statistical problem.
In general, language modelling is the task of predicting the next token in a sequence of tokens.

## Tokenization
We can define a token as the fundamental unit of a linguistic sequence, such as a word, a character, a syllable or any other meaningful unit. When converting a string of ASCII text to a sequence of tokens, we will use a tokenizer to split the string into tokens of our chosen granularity. The set of all possible tokens is called the vocabulary.
A simple character-level tokenizer might look like this:

```python
char_to_int = {'A': 1, 'a': 2, 'B': 3, 'b': 4, ...}
int_to_char = dict((v, k) for k, v in char_to_int.items()) # reverse mapping

def tokenize(text: str): List[int]:
    chars = []
    for c in text:
        chars.append(char_to_int[c])
    return chars
```

A word-level tokenizer might look like this:

```python
word_dict = {'apple': 1, ...}
int_to_word = dict((v, k) for k, v in word_dict.items()) # reverse mapping

def tokenize(text: str): List[int]:
    words = []
    for word in text.split(" "):
        words.append(word_dict[word])
    return words
```

In practice, we will use a tokenizer that has word-level, character-level and subword-level tokenization capabilities, choosing the granularity as course-grained as possible, as to "compress" the ASCII string into as few tokens as possible, while still retaining a lossless bidirectional mapping between the tokens and the original string.
The exact way in which we tokenize can have large implications on the performance of large language models on certain tasks.
Given that a token is the fundamental perceptive unit of a language model, sub-token perception is severely limited.
For example, GPT-3 will fail to understand the concept of syllable-counts and will thus fail at tasks such as counting the number of syllables in a word and or characters. Note however that eg. misspellings will force the tokenizer to choose a sub-token granularity that is more fine-grained than the word-level granularity, as to preserve the misspelling. Over a large corpus of text, a large language model will learn to equate the sub-token "misspelling" with the word-level token "misspelled", which can lead to an understanding of how certain common words are spelled, however, GPT-3 can be frequently observed to fail at spelling tasks and or syllable-dependent poetry generation tasks in English, while excelling at arguably more difficult tasks such as summarization, question answering and code generation.

The following figure shows ChatGPT, a variation of GPT-3.5 failing at generating a 5/7/5 haiku in English:
![chatgpt_haiku_fail](./figures/chatgpt_haiku_fail.png)

With a concrete definition of the unit of token, we can move on to modelling sequences of tokens.

## Language modelling as a statistical problem
Given a sequence of tokens $x_1, x_2, ..., x_n$, we want to predict the next token $x_{n+1}$.
Our language model will a discrete multinomial probability distribution over all tokens of the vocabulary, given the sequence of previous tokens. Iteratively sampling from this distribution will generate a sequence of tokens that is likely to occur language-wise.
This is a form of conditional probability, where the probability of the next token depends on the previous tokens.
One such approach is called a Markov chain. While Markov chains are a more general concept, that models the probability of an event occuring given a state of a system attained through previous events, we can specialize this approach to language modelling by assuming that the state of the system is formed through sequence of previous tokens.
When utilizing language models, we do not distinguish between a state attained through external (user) input and the state of the state attained to sampling from the language model.

# Bigram language modelling
The simplest form of language modelling is called unigram language modelling, where we assume that the probability of the next token depends only on the previous token.
This is a very naive assumption, as it does not take into account any context, but it is a good starting point for understanding the concept of language modelling.

We can define a bi-gram language model as a matrix $P$ of size $V \times V$, where $V$ is the size of the vocabulary.


# Language Modelling using NNs

# Language modeling using Transformers