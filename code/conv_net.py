import json
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch import optim


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.ndf = 20 * 8 * 8
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1, bias=False)
        self.fc1 = nn.Linear(self.ndf, 10, bias=False)
        # Can be exchanged with different activation functions
        self.act_func = nn.Sigmoid()

    def forward(self, x):
        act_func1 = self.act_func(self.conv1(x))
        x = self.pool(act_func1)
        x = self.act_func(self.conv2(x))
        act_func2 = x.clone()
        x = x.view(-1, self.ndf)
        x = self.fc1(x)
        return x, act_func1, act_func2

    def set_parameter(self, param_dict):
        """
        Sets the state of the network
        """
        st_dict = {}
        for key, value in param_dict.items():
            st_dict[key] = torch.nn.Parameter(
                torch.Tensor(value.cpu().float()))
        self.load_state_dict(st_dict)


def init_weights(m):
    """
    Initializes the network applying a normal distribution with given mean
    and std
    """
    mean = 0.
    if type(m) == nn.Linear:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.fill_(1)
    elif type(m) == nn.Conv2d:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.fill_(1)


def get_data(root, batch_size, device):
    """
    Returns MNIST training and testing datasets
    """
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    # Load data and normalize images to mean 0 and std 1
    # training set
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0], [1])
         ])
    train_loader_mnist = torch.utils.data.DataLoader(
        datasets.MNIST(root=root, train=True, download=True,
                       transform=transform),
        batch_size=batch_size, shuffle=True, **kwargs)
    # test set
    test_loader_mnist = torch.utils.data.DataLoader(
        datasets.MNIST(root=root, train=False,
                       transform=transform),
        batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader_mnist, test_loader_mnist


def train(epoch, train_loader_mnist, optimizer):
    net.train()
    train_loss = 0
    act_func = {'act1': [], 'act2': [], 'act1_mean': [], 'act2_mean': [],
                'act1_std': [], 'act2_std': [], 'act3': [], 'act3_mean': [],
                'act3_std': []}
    grads = {'conv1_grad': [], 'conv2_grad': [], 'fc1_grad': [],
             'conv1_grad_mean': [], 'conv2_grad_mean': [], 'fc1_grad_mean': [],
             'conv1_grad_std': [], 'conv2_grad_std': [], 'fc1_grad_std': []}
    for idx, (img, target) in enumerate(train_loader_mnist):
        optimizer.zero_grad()
        # network prediction for the image
        output, act1, act2 = net(img)
        act3 = F.softmax(output, dim=1)
        act_func['act1_mean'].append(act1.mean().item())
        act_func['act2_mean'].append(act2.mean().item())
        act_func['act3_mean'].append(act3.mean().item())
        act_func['act1_std'].append(act1.std().item())
        act_func['act2_std'].append(act2.std().item())
        act_func['act3_std'].append(act3.std().item())
        # calculate the loss
        loss = criterion(output, target)
        # backprop
        loss.backward()
        grads['conv1_grad_mean'].append(net.conv1.weight.grad.mean().item())
        grads['conv2_grad_mean'].append(net.conv2.weight.grad.mean().item())
        grads['fc1_grad_mean'].append(net.fc1.weight.grad.mean().item())
        grads['conv1_grad_std'].append(net.conv1.weight.grad.std().item())
        grads['conv2_grad_std'].append(net.conv2.weight.grad.std().item())
        grads['fc1_grad_std'].append(net.fc1.weight.grad.std().item())

        train_loss += loss.item()
        optimizer.step()

        if idx % 200 == 0:
            print('Loss {} in epoch {}, idx {}'.format(
                loss.item(), epoch, idx))
            grads['conv1_grad'].append(net.conv1.weight.grad.detach().numpy())
            grads['conv2_grad'].append(net.conv2.weight.grad.detach().numpy())
            grads['fc1_grad'].append(net.fc1.weight.grad.detach().numpy())
            act_func['act1'].append(act1.detach().numpy())
            act_func['act2'].append(act2.detach().numpy())
            act_func['act3'].append(act3.detach().numpy())
            # torch.save(net.state_dict(), 'results/model_it{}.pt'.format(idx))

    print('Average loss: {} epoch:{}'.format(
        train_loss / len(train_loader_mnist.dataset), epoch))
    if epoch % 10 == 0:
        opt_name = opt.__class__.__name__
        if not os.path.exists(opt_name):
            os.mkdir(opt_name)
        np.save(os.path.join(
            opt_name, 'gradients_ep{}.npy'.format(epoch)), grads)
        np.save(os.path.join(opt_name, 'act_func_ep{}.npy'.format(epoch)),
                act_func)


def test(epoch, test_loader_mnist):
    """
    Network evaluation, testing stage
    """
    net.eval()
    test_accuracy = 0
    test_loss = 0
    with torch.no_grad():
        for idx, (img, target) in enumerate(test_loader_mnist):
            output, _, _ = net(img)
            loss = criterion(output, target)
            test_loss += loss.item()
            # network prediction
            pred = output.argmax(1, keepdim=True)
            # how many image are correct classified, compare with targets
            test_accuracy += pred.eq(target.view_as(pred)).sum().item()

            if idx % 10 == 0:
                print('Test Loss {} in epoch {}, idx {}'.format(
                    loss.item(), epoch, idx))

        print('Test accuracy: {} Average test loss: {} epoch:{}'.format(
            100 * test_accuracy / len(test_loader_mnist.dataset),
            test_loss / len(test_loader_mnist.dataset), epoch))
    return 100 * test_accuracy / len(test_loader_mnist.dataset)


if __name__ == '__main__':
    # load configs
    with open('config.json', 'r') as jsfile:
        config = json.load(jsfile)
    seed = config['seed']
    torch.manual_seed(seed)
    net = ConvNet()
    accs = []
    # Cross entropy loss to calculate the loss
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = config['batch_size']
    std = 0
    epochs = config['epochs']
    root = config['root']
    train_loader, test_loader = get_data(root, batch, device)
    adam = optim.Adam(net.parameters(), lr=1e-3)
    sgd = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    optimizers = [sgd, adam]
    stds = [1]
    for opt in optimizers:
        for s in stds:
            std = s
            net.apply(init_weights)
            for ep in range(0, 1):
                train(ep, train_loader, opt)
                print('training done')
                acc = test(ep, test_loader)
                accs.append(acc)
    torch.save(accs, 'test_acc.pt')
