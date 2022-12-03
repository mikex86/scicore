from math import sqrt

import matplotlib.pyplot as plt
import torch
from tqdm import trange

from javarandom import Random
import torch.nn.functional as F

dataset_random = Random(123)
words = open('names.txt', 'r').read().splitlines()

# Shuffle conform with Java's Collections.shuffle() method
for i in range(len(words), 1, -1):
    j = dataset_random.next_int(i)
    copy = words[i - 1]
    words[i - 1] = words[j]
    words[j] = copy

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

# build the dataset
BLOCK_SIZE = 3  # context length: how many characters do we take to predict the next one?
VOCAB_SIZE = len(chars) + 1  # +1 for the dot
EMBEDDING_SIZE = 8
N_HIDDEN = 200


def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * BLOCK_SIZE
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # crop and append

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y


n1 = int(0.8 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xte, Yte = build_dataset(words[n1:])

random = Random(123)


def _init_linear_layer_like_java(param_tensor, output_size):
    k = 1 / sqrt(output_size)
    flat_cpy = param_tensor.detach().view(-1)
    for flat_idx in range(param_tensor.numel()):
        flat_cpy[flat_idx] = random.next_float() * 2 * k - k
    param_tensor.data = flat_cpy.reshape(param_tensor.shape)


def _init_gaussian_like_java(param_tensor):
    flat_cpy = param_tensor.detach().view(-1)
    for flat_idx in range(param_tensor.numel()):
        flat_cpy[flat_idx] = random.next_gaussian()
    param_tensor.data = flat_cpy.reshape(param_tensor.shape)


N_TRAINING_STEPS = 200000
BATCH_SIZE = 32
INITIAL_LEARNING_RATE = 0.1
END_LEARNING_RATE = 0.05
LEARNING_RATE_DECAY_FACTOR = (END_LEARNING_RATE / INITIAL_LEARNING_RATE) ** (1 / N_TRAINING_STEPS)

if __name__ == '__main__':
    # Declare parameters
    C = torch.zeros((VOCAB_SIZE, EMBEDDING_SIZE), dtype=torch.float32)
    W1 = torch.zeros((N_HIDDEN, BLOCK_SIZE * EMBEDDING_SIZE), dtype=torch.float32)
    b1 = torch.zeros(N_HIDDEN, dtype=torch.float32)
    W2 = torch.zeros((VOCAB_SIZE, N_HIDDEN), dtype=torch.float32)
    b2 = torch.zeros(VOCAB_SIZE, dtype=torch.float32)

    # Initialize parameters
    _init_gaussian_like_java(C)
    C -= 0.5
    _init_linear_layer_like_java(W1, W1.shape[1])
    _init_linear_layer_like_java(b1, W1.shape[1])
    _init_linear_layer_like_java(W2, W2.shape[1])
    _init_linear_layer_like_java(b2, W2.shape[1])

    # Collect parameters
    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True

    # Get loss on train set before training
    emb = C[Xtr]
    h = emb.view(-1, BLOCK_SIZE * EMBEDDING_SIZE) @ W1.T
    h = h + b1
    h = torch.tanh(h)
    logits = h @ W2.T + b2
    loss = F.cross_entropy(logits, Ytr)
    print("Loss on training set before training:", loss.item())

    # Start training

    lri = []
    lossi = []
    stepi = []

    start_idx = 0
    for step in trange(N_TRAINING_STEPS):
        # Construct Mini-batch

        # end_index = start_idx + BATCH_SIZE
        # if end_index >= Xtr.shape[0]:
        #     over_shoot = end_index - Xtr.shape[0]
        #     idx = torch.cat([
        #         torch.arange(start_idx, Xtr.shape[0]),
        #         torch.arange(0, over_shoot)
        #     ])
        #     start_idx = over_shoot
        # else:
        #     idx = torch.arange(start_idx, end_index)
        #     start_idx = end_index

        idx = torch.zeros(BATCH_SIZE, dtype=torch.long)
        for i in range(BATCH_SIZE):
            idx[i] = dataset_random.next_int(Xtr.shape[0])

        X = Xtr[idx]  # (batch_size, block_size)
        Y = Ytr[idx]  # (batch_size,)

        emb = C[X]  # (batch_size, block_size, embedding_size)
        emb_flat = emb.reshape(-1, BLOCK_SIZE * EMBEDDING_SIZE)  # (batch_size, block_size * embedding_size)

        # forward pass
        h = torch.tanh(emb_flat @ W1.T + b1)  # (batch_size, n_hidden)
        logits = h @ W2.T + b2  # (batch_size, vocab_size)
        loss = F.cross_entropy(logits, Y)  # (1,)

        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        lr = INITIAL_LEARNING_RATE * pow(LEARNING_RATE_DECAY_FACTOR, step)

        # update parameters
        for p in parameters:
            p.data -= lr * p.grad

        # track stats
        lossi.append(loss.log10().item())
        stepi.append(step)
        lri.append(lr)

    plt.plot(stepi, lossi)
    plt.show()
    plt.plot(stepi, lri)
    plt.show()

    # Get loss on train set after training
    emb = C[Xtr]
    h = torch.tanh(emb.view(-1, BLOCK_SIZE * EMBEDDING_SIZE) @ W1.T + b1)
    logits = h @ W2.T + b2
    loss = F.cross_entropy(logits, Ytr)
    print("Loss on training set:", loss.item())

    # visualize dimensions 0 and 1 of the embedding matrix C for all characters
    plt.figure(figsize=(8, 8))
    plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
    for i in range(C.shape[0]):
        plt.text(C[i, 0].item(), C[i, 1].item(), itos[i], ha="center", va="center", color='white')
    plt.grid('minor')
    plt.show()

    # Sample from the model
    sampling_rand = Random(123)

    for _ in range(20):
        out = []
        context = [0] * BLOCK_SIZE  # initialize with all ...

        while True:
            emb = C[torch.tensor([context])]  # (1,block_size,d)
            h = torch.tanh(emb.view(1, -1) @ W1.T + b1)
            logits = h @ W2.T + b2
            probs = F.softmax(logits, dim=1)

            # Multinomial sampling
            # shuffle probs (conform with Java's Collections .shuffle())
            indices = torch.arange(0, probs.shape[1]).view(probs.shape)
            for i in range(indices.shape[1], 1, -1):
                j = sampling_rand.next_int(i)
                copy = indices[0, i - 1].item()
                indices[0, i - 1] = indices[0, j].item()
                indices[0, j] = copy

            u = sampling_rand.next_float()
            cumulative_probs = 0.0
            ix = None
            for i in range(indices.shape[1]):
                cidx = indices[0, i].item()
                cumulative_probs += probs[0, cidx].item()
                if cumulative_probs >= u:
                    ix = cidx
                    break
            if ix is None:
                raise Exception("very bad")
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break

        print(''.join(itos[i] for i in out))

    print("Done")
