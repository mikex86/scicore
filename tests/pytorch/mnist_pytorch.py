from math import sqrt

import torch
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import trange
from javarandom import Random
import time


def save_tensor(tensor: torch.tensor, file_path: str):
    # Binary save format

    # data-type string(32) [eg. torch.float32]
    # n_dims: int64
    # shape: int64[n_dims]
    # tensor data

    content = tensor.detach().numpy().tobytes()
    with open(file_path, 'wb') as f:
        data_type_str = str(tensor.dtype)
        data_type_str_bytes_padded = data_type_str.encode('utf-8') + b'\0' * (32 - len(data_type_str))
        f.write(data_type_str_bytes_padded)
        shape = tensor.shape
        f.write(len(shape).to_bytes(8, 'big'))
        for dim in shape:
            f.write(dim.to_bytes(8, 'big'))
        f.write(content)


random = Random(123)


class MnistNet(torch.nn.Module):

    def __init__(self):
        super(MnistNet, self).__init__()
        self.act = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(28 * 28, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self._init_like_java(self.fc1.weight, self.fc1.weight.shape[1])
        self._init_like_java(self.fc1.bias, self.fc1.bias.shape[0])
        self._init_like_java(self.fc2.weight, self.fc2.weight.shape[1])
        self._init_like_java(self.fc2.bias, self.fc2.bias.shape[0])

    @staticmethod
    def _init_like_java(param_tensor, output_size):
        k = 1 / sqrt(output_size)
        flat_cpy = param_tensor.detach().view(-1)
        for flat_idx in range(param_tensor.numel()):
            flat_cpy[flat_idx] = random.next_float() * 2 * k - k
        param_tensor.data = flat_cpy.reshape(param_tensor.shape)

    def forward(self, x: torch.tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        # softmax
        exp = torch.exp(x)
        return exp / torch.sum(exp, dim=1, keepdim=True)


if __name__ == '__main__':
    transform = Compose([
        ToTensor()
    ])
    train_set = MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = MNIST(root='./data', train=False, download=True, transform=transform)

    batch_size = 32
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    net = MnistNet()

    n_steps = 60000
    lr = 0.01

    train_it = iter(train_loader)
    step_range = trange(n_steps, desc='Training')

    start_time = time.time()
    loss = None
    for i in step_range:
        # next from train_loader
        try:
            X, Y = next(train_it)
        except StopIteration:
            train_it = iter(train_loader)
            X, Y = next(train_it)

        Y_pred = net(X.view(-1, 28 * 28))
        Y_one_hot = torch.nn.functional.one_hot(Y, num_classes=10).float()
        loss = (Y_pred - Y_one_hot).pow(2).sum() / float(batch_size)

        loss.backward()

        step_range.set_description(f'Training loss: {loss.item():.5f}')

        with torch.no_grad():
            for param in net.parameters():
                param -= lr * param.grad
                param.grad.zero_()

    print(f'Final loss: {loss.item()}')
    print(f'Train time: {time.time() - start_time:.2f}s')

    # Test the network
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to('cpu')
            labels = labels.to('cpu')
            images = images.view(-1, 28 * 28)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {correct / total}")

