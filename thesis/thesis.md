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
The weight matrix is organized such that the weight at $w_jk$ is the weight of the connection from the $k$-th neuron of the previous layer to the $j$-th neuron of the current layer.

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

In this figure, we see a well-defined differentiable binary operator $f(x, y)$ which computes an output $z$. The output $z$ is an intermediary or the final value of the forward pass. To differentiate this operation, wee need the gradients "up until this operation" - the upstream gradients $\frac{\partial L}{\partial z}$. Then we proceed by computing the local gradients for both inputs $x$ and $y$. The local gradients only tell us how the output of $z$ is influenced by the input $x$ or $y$. The upstream gradients tell us how the output of $z$ is affects the loss function $L$. The product of the local and upstream gradients is the gradient of the loss function with respect to the input $x$ or $y$. Note that $x$ and $y$ can be functions themselves. In this case the gradients we computed are referred to as "downstream gradients" and become the "upstream gradients" for the next operation in the chain that computed the given input variable. 

Note that normally we will determine whether differentiating in respect to a given input of the operation is even necessary given what paramters we want to differentiate with respect to.
Software capable of differentiating such arbitrary graphs of operations is referred to as "Autograd engines" and are the backbone of modern deep learning frameworks.

### Scalar-level autograd
We will now explore implementing a simple scalar-based autograd engine. Note that this simple autograd engine is not part of SciCore, as this approach is not suitable for efficient differentiation of large neural networks.
Later, we will explore autograd where the "atoms" of differentiations are not the individual scalars, but rather the tensors that are the inputs to tensor-based operations.
The code for this section can be found on GitHub: https://github.com/mikex86/scalargrad

First, we will define a simple `Value` class to represent the scalar values that we will use as computational atoms.

```java
public class Value {
    private final double v;
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

The first operation we will implement is the multiplication operation. This operation will take two inputs and is thus a BinaryOperation, which redefines the `perform`
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

The second operation we will implement is the pow operation.

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
                // if we don't have a node for this value, it means that it is a value declaration
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

In the following code snippet, we recursively apply the chain rule by first computing the downstream gradients of the input nodes of the currently traversed operation and recursively ascending the graph to all nodes the operation depends on. Computed downstream gradients will become the upstream gradients of the next operation in the recursion.

```java
public class Graph {
    ...
    public void backward() {
        Value rootNodeValue = rootNode.getValue();
        rootNodeValue.accumulateGrad(1); // dL/dL = 1
        backward(rootNode);
    }

    private void backward(Node node) {
        if (node instanceof OperationNode operationNode) {
            Operation operation = operationNode.getOperation();
            List<Value> inputValues = new ArrayList<>();
            for (Node inputNode : operationNode.getInputNodes()) {
                inputValues.add(inputNode.getValue());
            }
            Value operationOutput = operationNode.getValue();
            operation.backward(operationOutput, inputValues);

            for (Node inputNode : operationNode.getInputNodes()) {
                backward(inputNode);
            }
        }
    }
    ...
}
```

Now we will implement the `backward` method for all of our operations:

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

```java
public class PowOp {
    ...
    @Override
    public void backward(Value c, Value a, Value b) {
        double upstreamGradient = c.getGrad();
        // Power rule: d/dx (x^y) = y * x^(y-1)
        a.accumulateGrad(upstreamGradient * b.getValue() * Math.pow(a.getValue(), b.getValue() - 1));
        // Exponentiation rule: d/dy (x^y) = x^y * ln(x)
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