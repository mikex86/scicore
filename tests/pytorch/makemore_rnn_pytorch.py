from typing import Tuple, Mapping

import torch
from torch.nn import Linear, Embedding
from torch.nn import Module, Parameter
from torch.utils.data import Dataset
from torch.nn import functional as F

EMBEDDING_SIZE = 32
HIDDEN_SIZE = 64
VOCAB_SIZE = 26 + 1  # +1 for start/end token


class NamesDataset(Dataset):

    def __init__(self, words, chars):
        self.words = words
        self.chars = chars
        self.max_word_len = max(len(w) for w in words)
        self.stoi = {s: i + 1 for i, s in enumerate(chars)}
        self.stoi['.'] = 0  # special start/end token
        self.itos = {i: s for s, i in self.stoi.items()}
        self.X, self.Y = self.__build_dataset()

    def __build_dataset(self) -> Tuple[torch.tensor, torch.tensor]:
        X, Y = [], []
        for w in self.words:
            x = [0] * (len(w) + 2)  # +2 for start and end tokens
            x[0] = self.stoi['.']
            x[1:] = [self.stoi[ch] for ch in w]
            x[-1] = self.stoi['.']
            x_tensor = torch.tensor(x + [0] * (self.max_word_len + 2 - len(x)))
            X.append(x_tensor)

            # shift all items in x by one to the right
            y = x[1:] + [-1]
            y_tensor = torch.tensor(y + [-1] * (self.max_word_len + 2 - len(y)))
            Y.append(y_tensor)
        return torch.stack(X), torch.stack(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class MakeMoreRNN(Module):

    def __init__(self):
        super(MakeMoreRNN, self).__init__()
        self.embedding = Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        # Deviation from Mikolov et al. 2010, initial hidden state is learned, not zero
        self.start = Parameter(torch.zeros(1, HIDDEN_SIZE, requires_grad=True))
        self.rnn_cell = Linear(in_features=EMBEDDING_SIZE + HIDDEN_SIZE, out_features=HIDDEN_SIZE)
        self.lm_head = Linear(in_features=HIDDEN_SIZE, out_features=VOCAB_SIZE)

    def forward(self, X: torch.Tensor) -> torch.tensor:
        emb = self.embedding(X)  # (batch_size, seq_len, embedding_size)
        hprev = self.start.expand(X.shape[0], -1)  # (batch_size, hidden_size)

        # Iterate over the sequence
        hiddens = []
        for i in range(X.shape[1]):
            embi = emb[:, i, :]  # (batch_size, embedding_size)
            xh = torch.cat((embi, hprev), dim=1)
            hprev = self.rnn_cell(xh)
            hiddens.append(hprev)

        hiddens = torch.stack(hiddens, dim=1)  # (batch_size, seq_len, hidden_size)
        logits = self.lm_head(hiddens)  # (batch_size, seq_len, vocab_size)
        return logits


N_PREDICTIONS = 10
PREDICTION_MAX_WORD_LEN = 20


@torch.no_grad()
@torch.inference_mode()
def make_predictions(model: MakeMoreRNN, itos: Mapping[int, str]):
    print('Predictions:')
    x = torch.zeros((N_PREDICTIONS, 1), dtype=torch.long)
    y_out = torch.zeros((N_PREDICTIONS, 1), dtype=torch.long)
    finished = torch.zeros(N_PREDICTIONS, dtype=torch.bool)
    for i in range(PREDICTION_MAX_WORD_LEN):
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)[..., -1, :]
        y = torch.multinomial(probs, num_samples=1)
        y_out = torch.cat((y_out, y), dim=1)
        x = y_out
        finished = finished | (y == 0)
        if finished.all():
            break

    words = []
    for i in range(N_PREDICTIONS):
        word = ''
        for j in range(y_out.shape[1] - 1):
            y_pred_char = itos[y_out[i, j + 1].item()]
            if y_pred_char == '.':
                break
            word += y_pred_char
        words.append(word)

    for word in words:
        print('\t' + word)
    print()


def create_datasets() -> Tuple[NamesDataset, NamesDataset]:
    with open('names.txt', 'r') as f:
        dataset_text = f.read()
    words = dataset_text.splitlines()
    chars = sorted(set(''.join(words)))
    train_size = int(len(words) * 0.8)
    train_words = words[:train_size]
    test_words = words[train_size:]
    train_dataset = NamesDataset(train_words, chars)
    test_dataset = NamesDataset(test_words, chars)
    return train_dataset, test_dataset


BATCH_SIZE = 32
N_TRAINING_STEPS = 200_000


def main():
    train_dataset, test_dataset = create_datasets()

    model = MakeMoreRNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)

    for step in range(N_TRAINING_STEPS):
        idx = torch.randint(0, len(train_dataset), (BATCH_SIZE,))
        X, Y = train_dataset[idx]

        logits = model(X)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), Y.view(-1), ignore_index=-1)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f'Step {step}: loss = {loss.item()}')

        if (step + 1) % 1001 == 0:
            make_predictions(model, train_dataset.itos)

    torch.save(model.state_dict(), 'makemore_rnn.pt')


if __name__ == '__main__':
    main()
