from conv_net import ConvNet
from enkf_pytorch import EnsembleKalmanFilter as EnKF
from enkf_pytorch import _encode_targets
from numpy.linalg import norm

import copy
import json
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('config.json', 'r') as jsfile:
    config = json.load(jsfile)


class DataLoader:
    """
    Convenience class.
    """

    def __init__(self):
        self.data_mnist = None
        self.test_mnist = None
        self.data_mnist_loader = None
        self.test_mnist_loader = None

    def init_iterators(self, root, batch_size):
        self.data_mnist_loader, self.test_mnist_loader = self.load_data(
            root, batch_size)
        # here are the expensive operations
        self.data_mnist = self._make_iterator(self.data_mnist_loader)
        self.test_mnist = self._make_iterator(self.test_mnist_loader)

    @staticmethod
    def _make_iterator(iterable):
        """
        Return an iterator for a given iterable.
        Here `iterable` is a pytorch `DataLoader`
        """
        return iter(iterable)

    @staticmethod
    def load_data(root, batch_size):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.], [1.])
             ])
        # load now MNIST dataset
        trainset_mnist = torchvision.datasets.MNIST(root=root,
                                                    train=True,
                                                    download=True,
                                                    transform=transform)
        trainloader_mnist = torch.utils.data.DataLoader(trainset_mnist,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        num_workers=2)

        testset_mnist = torchvision.datasets.MNIST(root=root,
                                                   train=False,
                                                   download=True,
                                                   transform=transform)
        testload_mnist = torch.utils.data.DataLoader(testset_mnist,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     num_workers=2)
        return trainloader_mnist, testload_mnist

    def dataiter_mnist(self):
        """ MNIST training set list iterator """
        return self.data_mnist.next()

    def testiter_mnist(self):
        """ MNIST test set list iterator """
        return self.test_mnist.next()


class MnistOptimizee(torch.nn.Module):
    def __init__(self, n_ensembles, batch_size, root):
        super(MnistOptimizee, self).__init__()

        self.n_ensembles = n_ensembles
        self.batch_size = batch_size
        self.root = root
        self.length = 0

        self.conv_net = ConvNet().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.data_loader = DataLoader()
        self.data_loader.init_iterators(self.root, self.batch_size)
        self.dataiter_mnist = self.data_loader.dataiter_mnist
        self.testiter_mnist = self.data_loader.testiter_mnist

        self.generation = 0   # iterations
        self.inputs, self.labels = self.dataiter_mnist()
        self.inputs = self.inputs.to(device)
        self.labels = self.labels.to(device)
        self.test_input, self.test_label = self.testiter_mnist()
        self.test_input = self.test_input.to(device)
        self.test_label = self.test_label.to(device)

        # For plotting
        self.train_pred = []
        self.test_pred = []
        self.train_acc = []
        self.test_acc = []
        self.train_cost = []
        self.test_cost = []
        self.train_loss = []
        self.test_loss = []
        self.targets = []
        self.output_activity_train = []
        self.output_activity_test = []
        self.act_func = {'act1': [], 'act2': [], 'act1_mean': [],
                         'act2_mean': [], 'act1_std': [], 'act2_std': [],
                         'act3': [], 'act3_mean': [], 'act3_std': []}
        self.test_act_func = {'act1': [], 'act2': [], 'act1_mean': [],
                              'act2_mean': [], 'act1_std': [], 'act2_std': [],
                              'act3': [], 'act3_mean': [], 'act3_std': []}

        self.targets.append(self.labels.cpu().numpy())

        for key in self.conv_net.state_dict().keys():
            self.length += self.conv_net.state_dict()[key].nelement()

    def create_individual(self):
        # get weights, biases from networks and flatten them
        # convolutional network parameters
        conv_ensembles = []
        sigma = config['sigma']
        with torch.no_grad():
            for _ in range(self.n_ensembles):
                conv_ensembles.append(np.random.normal(0, sigma,
                                                       size=self.length))
            return dict(conv_params=torch.as_tensor(np.array(conv_ensembles),
                                                    device=device),
                        targets=self.labels,
                        input=self.inputs.squeeze())

    def load_model(self, path='conv_params.npy'):
        print('Loading model from path: {}'.format(path))
        conv_params = np.load(path).item()
        conv_ensembles = conv_params.get('ensemble')
        return dict(conv_params=torch.as_tensor(np.array(conv_ensembles),
                                                device=device),
                    targets=self.labels,
                    input=self.inputs.squeeze())

    def set_parameters(self, ensembles):
        # set the new parameter for the network
        conv_params = ensembles.mean(0)
        ds = self._shape_parameter_to_conv_net(conv_params)
        self.conv_net.set_parameter(ds)
        print('---- Train -----')
        print('Iteration ', self.generation)
        generation_change = config['repetitions']
        with torch.no_grad():
            inputs = self.inputs.to(device)
            labels = self.labels.to(device)

            if self.generation % generation_change == 0:
                self.inputs, self.labels = self.dataiter_mnist()
                self.inputs = self.inputs.to(device)
                self.labels = self.labels.to(device)
                print('New MNIST set used at generation {}'.format(
                    self.generation))
                # append the outputs
                self.targets.append(self.labels.cpu().numpy())
            # get network predicitions
            outputs, act1, act2 = self.conv_net(inputs)
            act3 = outputs
            # save all important calculations
            self.act_func['act1'] = act1.cpu().numpy()
            self.act_func['act2'] = act2.cpu().numpy()
            self.act_func['act3'] = act3.cpu().numpy()
            self.act_func['act1_mean'].append(act1.mean().item())
            self.act_func['act2_mean'].append(act2.mean().item())
            self.act_func['act3_mean'].append(act3.mean().item())
            self.act_func['act1_std'].append(act1.std().item())
            self.act_func['act2_std'].append(act2.std().item())
            self.act_func['act3_std'].append(act3.std().item())
            self.output_activity_train.append(
                F.softmax(outputs, dim=1).cpu().numpy())
            conv_loss = self.criterion(outputs, labels).item()
            self.train_loss.append(conv_loss)
            train_cost = _calculate_cost(_encode_targets(labels, 10),
                                         F.softmax(outputs, dim=1).cpu().numpy(), 'MSE')
            train_acc = score(labels.cpu().numpy(),
                              np.argmax(F.softmax(outputs, dim=1).cpu().numpy(), 1))
            print('Cost: ', train_cost)
            print('Accuracy: ', train_acc)
            print('Loss:', conv_loss)
            self.train_cost.append(train_cost)
            self.train_acc.append(train_acc)
            self.train_pred.append(
                np.argmax(F.softmax(outputs, dim=1).cpu().numpy(), 1))

            print('---- Test -----')
            test_output, act1, act2 = self.conv_net(self.test_input)
            test_loss = self.criterion(test_output, self.test_label).item()
            self.test_act_func['act1'] = act1.cpu().numpy()
            self.test_act_func['act2'] = act2.cpu().numpy()
            self.test_act_func['act1_mean'].append(act1.mean().item())
            self.test_act_func['act2_mean'].append(act2.mean().item())
            self.test_act_func['act3_mean'].append(test_output.mean().item())
            self.test_act_func['act1_std'].append(act1.std().item())
            self.test_act_func['act2_std'].append(act2.std().item())
            self.test_act_func['act3_std'].append(test_output.std().item())
            test_output = test_output.cpu().numpy()
            self.test_act_func['act3'] = test_output
            test_acc = score(self.test_label.cpu().numpy(),
                             np.argmax(test_output, 1))
            test_cost = _calculate_cost(_encode_targets(self.test_label.cpu().numpy(), 10),
                                        test_output, 'MSE')
            print('Test accuracy', test_acc)
            print('Test loss: {}'.format(test_loss))
            self.test_acc.append(test_acc)
            self.test_pred.append(np.argmax(test_output, 1))
            self.test_cost.append(test_cost)
            self.output_activity_test.append(test_output)
            self.test_loss.append(test_loss)
            print('-----------------')
            conv_params = []
            for idx, c in enumerate(ensembles):
                ds = self._shape_parameter_to_conv_net(c)
                self.conv_net.set_parameter(ds)
                params, _, _ = self.conv_net(inputs)
                conv_params.append(params.t().cpu().numpy())

            outs = {
                'conv_params': torch.tensor(conv_params).to(device),
                'conv_loss': float(conv_loss),
                'input': self.inputs.squeeze(),
                'targets': self.labels
            }
        return outs

    def _shape_parameter_to_conv_net(self, params):
        """
        Create a dictionary which includes the shapes of the network
        architecture per layer
        """
        param_dict = dict()
        start = 0
        for key in self.conv_net.state_dict().keys():
            shape = self.conv_net.state_dict()[key].shape
            length = self.conv_net.state_dict()[key].nelement()
            end = start + length
            param_dict[key] = params[start:end].reshape(shape)
            start = end
        return param_dict


def _calculate_cost(y, y_hat, loss_function='CE'):
    """
    Loss functions
    :param y: target
    :param y_hat: calculated output (here G(u) or feed-forword output a)
    :param loss_function: name of the loss function
           `MAE` is the Mean Absolute Error or l1 - loss
           `MSE` is the Mean Squared Error or l2 - loss
           `CE` cross-entropy loss, requires `y_hat` to be in [0, 1]
           `norm` norm-2 or Forbenius norm of `y - y_hat`
    :return: cost calculated according to `loss_function`
    """
    if loss_function == 'CE':
        term1 = -y * np.log(y_hat)
        term2 = (1 - y) * np.log(1 - y_hat)
        return np.sum(term1 - term2)
    elif loss_function == 'MAE':
        return np.sum(np.absolute(y_hat - y)) / len(y)
    elif loss_function == 'MSE':
        return np.sum((y_hat - y) ** 2) / len(y)
    elif loss_function == 'norm':
        return norm(y - y_hat)
    else:
        raise KeyError(
            'Loss Function \'{}\' not understood.'.format(loss_function))


def score(x, y):
    """
    Calculates the accuracy between predicition and targets

    :param x: targets
    :param y: prediction
    :return: accuracy
    """
    print('target ', x)
    print('predict ', y)
    n_correct = np.count_nonzero(y == x)
    n_total = len(y)
    sc = n_correct / n_total
    return sc


def test(net, iteration, test_loader_mnist, criterion):
    """
    Test function with images from the test set
    """
    net.eval()
    test_accuracy = 0
    test_loss = 0
    with torch.no_grad():
        for idx, (img, target) in enumerate(test_loader_mnist):
            img = img.to(device)
            target = target.to(device)
            output, _, _ = net(img)
            loss = criterion(output, target)
            test_loss += loss.item()
            # network prediction
            pred = output.argmax(1, keepdim=True)
            # how many image are correct classified, compare with targets
            test_accuracy += pred.eq(target.view_as(pred)).sum().item()

            if idx % 10 == 0:
                print('Test Loss {} in iteration {}, idx {}'.format(
                    loss.item(), iteration, idx))
        ta = 100 * test_accuracy / len(test_loader_mnist.dataset)
        tl = test_loss / len(test_loader_mnist.dataset)
        print('------ Evaluation -----')
        print('Test accuracy: {} Average test loss: {}'.format(ta, tl))
        print('-----------------------')
    return ta, tl


def test_dynamic_change(ta, eps):
    """
    Test if `test_accuracy_i` is better than `test_accuracy_i-1`
    :param ta: test accuracy list of actual and previous run
    """
    if ta[-1] > eps:
        if ta[-1] >= ta[-2]:
            return True
        elif ta[-1] < ta[-2]:
            return False
    return -1


def save_dynamic_changes(ta, n_ensembles, reps, it, rng, dyn_c):
    dyn_c['model_reps'].append(reps)
    dyn_c['n_ensembles'].append(n_ensembles)
    dyn_c['iteration'].append(it)
    dyn_c['test_loss'].append((ta[-2], ta[-1]))
    dyn_c['rng'].append(rng)
    return dyn_c


if __name__ == '__main__':
    # path to data
    root = config['root']
    n_ensembles = config['n_ensembles']
    conv_loss_mnist = []
    # save test losses
    test_losses = []
    act_func = {}
    # set the seed for numpy and torch
    seed = config['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    batch_size = config['batch_size']
    # for dynamic change
    dynamic_change = config['dynamic_change']
    eps = config['eps']  # epsilon
    reps = config['reps']  # repetition
    lower_bound = config['lower_bound']
    upper_bound = config['upper_bound']
    dyn_change = {'model_reps': [], 'n_ensembles': [], 'iteration': [],
                  'rng': 60000, 'test_loss': []}

    # init model
    model = MnistOptimizee(root=root, batch_size=batch_size,
                           n_ensembles=n_ensembles).to(device)
    conv_ens = None
    g_scaler = config['gamma']
    gamma = np.eye(10) * g_scaler
    enkf = EnKF(maxit=1,
                online=False,
                n_batches=1)
    # len dataset divided by batch size and repetitions
    rng = int(60000 / batch_size * 8)
    i = 0
    while i < rng:
        model.generation = i + 1
        if i == 0:
            try:
                out = model.load_model('')
            except FileNotFoundError as fe:
                print(fe)
                print('Model not found! Initalizaing new ensembles.')
                out = model.create_individual()
            conv_ens = out['conv_params']
            out = model.set_parameters(conv_ens)
            print('loss {} generation {}'.format(out['conv_loss'],
                                                 model.generation))
        enkf.fit(data=out['input'],
                 ensemble=conv_ens,
                 ensemble_size=n_ensembles,
                 moments1=conv_ens.mean(0),
                 observations=out['targets'],
                 model_output=out['conv_params'], noise=0.0,
                 gamma=gamma)
        conv_ens = enkf.ensemble
        out = model.set_parameters(conv_ens)
        conv_loss_mnist.append(out['conv_loss'])
        if i % config['checkpoints'] == 0:
            # do some checkpoiting
            print('Checkpointing at iteration {}'.format(i), flush=True)
            param_dict = {
                'train_pred': model.train_pred,
                'test_pred': model.test_pred,
                'train_acc': model.train_acc,
                'test_acc': model.test_acc,
                'train_cost': model.train_cost,
                'test_cost': model.test_cost,
                'train_targets': model.targets,
                'train_act': model.output_activity_train,
                'test_act': model.output_activity_test,
                'test_targets': model.test_label.cpu().numpy(),
                'ensemble': conv_ens.cpu().numpy(),
                'act_func': model.act_func,
                'test_act_func': model.test_act_func,
                'train_loss': model.train_loss,
                'test_loss': model.test_loss,
            }
            torch.save(param_dict, 'conv_params_{}.pt'.format(i))
            act_func[str(i)] = {'train_act': model.act_func,
                                'test_act': model.test_act_func}
            test_losses.append(test(model.conv_net, i, model.data_loader.test_mnist_loader,
                                    nn.CrossEntropyLoss(reduction='sum')))
            torch.save(test_losses, 'test_losses_{}.pt'.format(i))
            # Test for dynamic changes
            if i > 0 and dynamic_change:
                tdc = test_dynamic_change(test_losses, eps)
                if tdc:
                    diff = test_losses[-1] - test_losses[-2]
                    # reduce n_ensembles by 1000, same for the model when the
                    # difference it not that big
                    if n_ensembles > lower_bound and diff <= 1:
                        n_ensembles -= 1000
                        enkf.ensemble_size = n_ensembles
                        model.n_ensembles = n_ensembles
                    # reduce model generation change
                    if reps > 0:
                        reps -= 1
                        model.generation_change = reps
                        # change also loop range
                        rng = 60000 / batch_size * reps
                    # Save changes in dictionary
                    dyn_change = save_dynamic_changes(ta=test_losses,
                                                      n_ensembles=n_ensembles,
                                                      reps=reps, it=i, rng=rng,
                                                      dyn_c=dyn_change)
                elif tdc != -1 and tdc is False:
                    diff = test_losses[-2] - test_losses[-1]
                    # increase n_ensembles by 1000, same for the model when
                    # the difference bigger than 1.
                    if n_ensembles < upper_bound and diff > 1.:
                        n_ensembles += 1000
                    enkf.ensemble_size = n_ensembles
                    model.n_ensembles = n_ensembles
                    # increase model generation change
                    if model.generation_change < 8:
                        reps += 1
                        model.generation_change = reps
                        # change also loop range
                        rng = 60000 / batch_size * reps
                    # Save changes in dictionary
                    dyn_change = save_dynamic_changes(ta=test_losses,
                                                      n_ensembles=n_ensembles,
                                                      reps=reps, it=i, rng=rng,
                                                      dyn_c=dyn_change)
                print('Dynamic changes: ', dyn_change)

        # Increase index
        i += 1

    param_dict = {
        'train_pred': model.train_pred,
        'test_pred': model.test_pred,
        'train_acc': model.train_acc,
        'test_acc': model.test_acc,
        'train_cost': model.train_cost,
        'test_cost': model.test_cost,
        'train_targets': model.targets,
        'train_act': model.output_activity_train,
        'test_act': model.output_activity_test,
        'test_targets': model.test_label.cpu().numpy(),
        'ensemble': conv_ens.cpu().numpy(),
        'act_func': model.act_func,
        'test_act_func': model.test_act_func,
        'train_loss': model.train_loss,
        'test_loss': model.test_loss,
    }
    torch.save(param_dict, 'conv_params.pt')
    act_func[str(i)] = {'train_act': copy.deepcopy(model.act_func),
                        'test_act': copy.deepcopy(model.test_act_func)}
    torch.save(act_func, 'act_func.pt')
    torch.save(test_losses, 'test_losses.pt')
