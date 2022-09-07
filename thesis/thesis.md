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

# Introduction

In this chapter, the problem of conversational AI is introduced and expressed as a mathematical problem.
Concepts will be introduced from first principles and the layers of abstraction traversed from the low-level mathematical foundations to the high-level concept that is a conversational AI system.

To start, the concept of a tensor will be introduced, which is a generalisation of a vector and a matrix and is the fundamental data structure in deep learning.
A definition for 'learning' in the machine learning context will be layed out, and the concept of a neural network will be introduced as an example of a system learning in such fashion.
While neural networks in their raw form of sufficient complexity are considered universal function approximators, architectures with intent to speed up the learning process in light of the task-specific problem structure are introduced.
The concept of a recurrent neural network will be introduced as an example of such an architecture, as a way to handle sequential data, while also focusing on their shortcomings.
The concept of a transformer will be introduced as a way to overcome the shortcomings of recurrent neural networks.
Lastly, we will introduce the concept of language models, which are a special case of transformers, as well the concept of a chatbot, which is a special case of language models.

# Definition of 'Learning'

Learning is a concept that is used in many different contexts, but in the context of machine learning, it is defined as the process of approximating an unknown function that maps an input to an output.
This approximation will be derived from a set of examples called the training data, where each example $(X, Y)$ is a pair of an input $X$, also called the feature vector, and an output $Y$, also referred to as the label.
The goal of the learning process is to find a function $f(X)$, also known as the model $m(X)$, that maps the input $X$ to the output $Y$, but specifically a function that will be able to map inputs $X$ to outputs $Y$ in a generalizing fashion, meaning that the
function will be able to map inputs $X$ to the correct outputs $Y$ that were not part of the set of examples used to derive the function.
Thus, from now on, we will refer to the term generalization as the predictive capability of the model $m(X)$ beyond the set of examples used to derive the function as well as the predictive capability of the function $m(X)$
for inputs $X$ outside the proximity of the set of examples used to derive the function. While for problems of low complexity, 'proximity' in this context can be equated to distance metrics such as the Euclidean distance,
an arbitrarily complex problem $P(X)$ has no inherent distance metric, as whether two inputs $(X1, X2)$ are considered similar in the context of the problem is determined by the nature of the problem itself.
But even if we were to have an idealized model $m_i(X)$ of the problem $P(X)$, and defined the proximity of two inputs as let's say $d(X1, X2) = |m_i(X1) - m_i(X2)|$,
we see that not only could the output $Y$ of $P(X)$ exist in the input domain of another problem $P_{2}(X)$ requiring another idealized model to determine how similar those outputs are - making it a recursive problem (with what exit condition?) -
but also that the similarity of inputs cannot necessarily be determined only by its effect on the output alone, as inputs $(X1, X2)$ might result in the same output $Y$, but for vastly different reasons, a reasonable interpreter of the problem $P(X)$ would be able to state.
As part of correctly generalizing must also mean not jumping to false conclusions, as the model might output the correct answer 'for the wrong reasons', we shall also incorporate this phenomenon into our definition of generalizing.

This seems to suggest we would need have access to some intermediate state as the model is evaluating the input $X$ in order to determine the similarity of the inputs, which is impossible given we defined the idealized model of the problem $P(X)$ as a black box.
This is analogous to asking a reasonable interpreter of the problem $P(X)$ to explain their thought process, which is again problematic because it requires another reasonable interpreter of the problem $P(X)$ to interpret whether the explanation is correct.
This also seems to suggest that the idealized model $m_i(X)$ is impossible to derive from input output pairs $(X, Y)$, as for an arbitrarily complex problem $P(X)$, similarly to what we established above, $\frac{\partial m(X, W) - Y}{\partial W}$ as a change to $W$
where $W$ are parameters of a sufficiently complex function $m(X)$ can mislead the model 'into the wrong direction', if the example when looked at in a vacuum seems to imply a pattern that does not hold in the set of potential $(X, Y)$ pairs in the problem domain of $P(X)$.
Generally, the only way to correct for a misleading example is for another example pair to compensate for the error in the previous pair, which is not guaranteed to happen. Thus deriving the idealized model $m_i(X)$ is only possible with an infinite amount of input output pairs $(X, Y)$,
Nevertheless, we could define a way to define the parameters $W_i$ of the idealized model $m_i(X)$ in the following fashion:

$$
W_{i} = \int_{1}^\infty \frac{\partial |m_i(X_n, W_n) - Y_n|}{\partial W_n} \space dn \space \text{with} \space W_0 \in \mathbb{R}

$$

where $W_n$ are the parameters of the model $m_i(X)$ at the $n$-th iteration of the learning process, and $X_n$ and $Y_n$ are the $n$-th input output pair $(X, Y)$ of the set of possible examples in the problem domain of $P(X)$.
This allows us to make basic statements about 'learnability'.
For example, a problem $P(X)$ where the number of examples that suggest wrong patterns is greater than the number of examples which suggest the correct pattern due to the innate nature of the problem such that $n_{confusing}>n_{clear}$,
the problem is fundamentally un-learnable.
