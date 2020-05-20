from conv_net import ConvNet
from torchvision import datasets, transforms
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch

sns.set(style="white")
sns.set_color_codes("dark")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.})


def plot_test_error(accuracies, iters):
    """
    Plot the test error for the Enkf method

    """
    fig, ax1 = plt.subplots()
    acc = 100 - np.array(accuracies)
    ax1.plot(acc, '.-', markersize=6., label=r'EnKF')
    ax1.set_ylabel('Test error in %')
    ax1.set_xlabel('Iterations')
    plt.xticks(range(len(iters)), iters)
    tkl = ax1.xaxis.get_ticklabels()
    for label in tkl[::2]:
        label.set_visible(False)
    plt.tight_layout()
    plt.legend()
    plt.savefig('enkf_test_error.pdf',
                bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_error_sgd_adam(accuracies):
    """
    Plots test errors for different realizations of sigma ($\sigma=1, \sigma=3$)
    when the network is initialized and optimized with SGD and Adam.
    """
    accs = np.array(accuracies)
    plt.plot(100 - accs[:49][:, 1].astype(float), label=r'SGD $\sigma=1$')
    plt.plot(100 - accs[49:98][:, 1].astype(float), label=r'SGD $\sigma=3$')
    plt.plot(100 - accs[99:147][:, 1].astype(float), label=r'ADAM $\sigma=1$')
    plt.plot(100 - accs[147:][:, 1].astype(float), label=r'ADAM $\sigma=3$')
    plt.xlabel('Epochs')
    plt.ylabel('Test error in %')
    plt.legend()
    # plt.xticks(range(0, 50, 10), range(1, 51, 10))
    plt.savefig('sgd_adam_test_error.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_different_accuracies(iters):
    """
    Plots accuracies when 500, 5000 and 10000 ensembles are used with EnKF.
    """
    norm_loss = 100 - np.array(torch.load('acc_loss.pt')[0])
    more_loss = 100 - np.array(torch.load('more_ensembles_acc_loss.pt')[0])
    less_loss = 100 - np.array(torch.load('less_ensembles_acc_loss.pt')[0])
    fig, ax1 = plt.subplots()
    ax1.plot(less_loss, '.-', label='100 ensembles')
    ax1.plot(norm_loss, '.-', label='5000 ensembles')
    ax1.plot(more_loss, '.-', label='10000 ensembles')
    ax1.set_ylabel('Test Error in %')
    ax1.set_xlabel('Iterations')
    plt.xticks(range(len(iters)), iters)
    tkl = ax1.xaxis.get_ticklabels()
    for label in tkl[::2]:
        label.set_visible(False)
    plt.legend()
    plt.show()
    fig.savefig('ensembles_diff_test_accuracies.pdf')


def plot_act_func_accuracies(iters):
    norm_loss = 100 - np.array(torch.load('acc_loss.pt')[0])
    more_loss = 100 - np.array(torch.load('relu_acc_loss.pt')[0])
    less_loss = 100 - np.array(torch.load('tanh_acc_loss.pt')[0])
    fig, ax1 = plt.subplots(figsize=(6, 6))
    ax1.plot(norm_loss, '.-', label='Logistic Function')
    ax1.plot(more_loss, '.-', label='ReLU')
    ax1.plot(less_loss, '.-', label='Tanh')
    ax1.set_ylabel('Test Error in %')
    ax1.set_xlabel('Iterations')
    plt.xticks(range(len(iters)), iters)
    tkl = ax1.xaxis.get_ticklabels()
    for label in tkl[::2]:
        label.set_visible(False)
    # ax1.set_title()
    # plt.tight_layout()
    plt.legend()
    plt.show()
    fig.savefig('act_func_test_accuracies.pdf',
                bbox_inches='tight',   pad_inches=0.1)


def adaptive_test_loss_splitted(normal_loss, dynamic_loss):
    adaptive_ta = 100 - np.unique(dynamic_loss.get('test_loss'))
    iterations = dynamic_loss.get('iteration')
    iterations.insert(0, 0)
    n_ens = dynamic_loss.get('n_ensembles')
    n_ens.insert(0, 5000)
    reps = dynamic_loss.get('model_reps')
    reps.insert(0, 8)
    normal_loss = 100 - np.array(normal_loss[0][:len(adaptive_ta)])
    # prepare plots
    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, sharex=True, figsize=(8.5, 4))
    ax3 = ax2.twinx()
    # plot 1
    marker = 'o'
    p11, = ax1.plot(adaptive_ta, marker=marker, label='fixed')
    p12, = ax1.plot(normal_loss, marker=marker, label='adaptive')
    ax1.set_xticks(range(len(iterations)))
    ax1.set_xticklabels(iterations)
    ax1.set_ylabel('Test Error  in %')
    ax1.set_xlabel('Iterations')
    # plot 2
    marker = 's--'
    markersize = 6
    # plot for n_ens normal
    p21, = ax2.plot(range(len(n_ens)), [5000] *
                    len(n_ens), marker, markersize=markersize)
    # plot for n_ens dynamic
    p22, = ax2.plot(n_ens, marker, markersize=markersize)
    # empty fake plot for the correct label color
    ax2p = ax2.plot([], marker, label='# ensembles', c='k')
    ax2.set_ylabel('Number of ensembles')
    # plot 3
    marker = '*--'
    markersize = 8
    # plot for reps normal
    p31, = ax3.plot(range(len(reps)), [8]*len(reps), marker,
                    c=p21.get_color(), ms=markersize)
    # plot for reps dynamic
    p32, = ax3.plot(range(len(reps)), reps, marker,
                    c=p22.get_color(), ms=markersize)
    # empty fake plot for the correct label color
    ax3p = ax3.plot([], marker, label='repetitions', c='k', ms=markersize)
    ax3.set_xticks(range(len(reps)))
    # ax3.tick_params(axis='y')
    ax3.set_ylabel('Number of repetitions')

    # merge labels into one legend
    lgd = ax2p + ax3p
    labs = [l.get_label() for l in lgd]
    ax1.legend(prop={'size': 10})
    ax2.legend(lgd, labs, prop={'size': 10})
    # remove ax ticks from second plot
    ax2.tick_params(axis=u'both', which=u'both', length=0)
    ax3.yaxis.set_tick_params(length=0)
    fig.tight_layout()
    fig.subplots_adjust(top=0.942, bottom=0.166, left=0.095, right=0.918,
                        hspace=0.1, wspace=0.388)
    fig.savefig('dynamic_changes_splitted.pdf', format='pdf',
                bbox_inches='tight')
    plt.show()


def plot_accuracy_all_iteration_std(startswith, inset=True, path='.'):
    fig, ax1 = plt.subplots()
    for starts in startswith:
        if starts.startswith('SGD'):
            mu_sgd, std_sgd = _load_pt_files(starts, path)
            mu_sgd = np.insert(mu_sgd, 0, 0)
            std_sgd = np.insert(std_sgd, 0, 0)
            mu_sgd = 100 - mu_sgd
        elif starts.startswith('acc'):
            mu_enkf, std_enkf = _load_pt_files(starts, path)
            mu_enkf = np.insert(mu_enkf, 0, 0)
            std_enkf = np.insert(std_enkf, 0, 0)
            mu_enkf = 100 - np.array(mu_enkf)

    # SGD
    ax1.plot(mu_sgd[: len(mu_enkf)], 'o-', label='SGD')
    lower_bound = (mu_sgd - std_sgd)[: len(mu_enkf)]
    upper_bound = (mu_sgd + std_sgd)[: len(mu_enkf)]
    ax1.fill_between(range(len(mu_enkf)), lower_bound, upper_bound, alpha=.3)
    # Enkf
    p3, = ax1.plot(mu_enkf, 'o-', label='EnKF')
    lower_bound = (mu_enkf - std_enkf)
    upper_bound = (mu_enkf + std_enkf)
    ax1.fill_between(range(len(mu_enkf)), lower_bound, upper_bound, alpha=.3)

    ax1.set_ylabel('Test Error in %')
    ax1.set_xlabel('Iteration')
    iters = range(0, 8000, 500)
    ax1.set_xticks(range(len(iters)))
    ax1.set_xticklabels(iters)
    plt.xlim(0, len(iters))
    plt.legend(prop={'size': 11}, loc='best')
    tkl = ax1.xaxis.get_ticklabels()
    if inset:
        _plot_std_zoomed_inset(ax1, p3, mu_enkf, lower_bound, upper_bound)
    for label in tkl[:: 2]:
        label.set_visible(False)
    plt.savefig('all_test_error_iteration.pdf',
                bbox_inches='tight', pad_inches=0.1)
    plt.show()


def _plot_std_zoomed_inset(ax, pl, mu, lower_bound, upper_bound):
    axins = zoomed_inset_axes(
        ax, zoom=5., bbox_to_anchor=(0.4, 0.4),
        bbox_transform=plt.gcf().transFigure)
    axins.plot(mu, 'o-', c=pl.get_color())
    axins.fill_between(range(len(mu)), lower_bound,
                       upper_bound, alpha=.3, color=pl.get_color())
    plt.setp(axins.get_xticklabels(), visible=False)
    plt.setp(axins.get_yticklabels(), visible=False)
    axins.set_xlim(5, 6.)
    axins.set_ylim(4, 5.5)
    mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5")


def _load_pt_files(startswith, path='.'):
    """
    Loads the pt files which should start with `acc` (EnKF) `SGD` and end with `.pt`
    Returns mean test_accuracies and standard deviations as np.arrays
    """
    files = []
    for file in os.listdir(path):
        if file.startswith(startswith) and file.endswith('.pt'):
            if file.startswith('acc'):
                f = os.path.join(path, file)
                files.append(np.array(torch.load(f))[:, 0])
            else:
                f = os.path.join(path, file)
                files.append(torch.load(f))
    files = np.array(files)
    return files.mean(0), files.std(0)


def test(net, test_loader_mnist):
    """
    Network eval function
    """
    net.eval()
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():
        for idx, (img, target) in enumerate(test_loader_mnist):
            output, _, _ = net(img)
            loss = criterion(output, target)
            test_loss += loss.item()
            # network prediction
            pred = output.argmax(1, keepdim=True)
            # how many image are correct classified, compare with targets
            ta = pred.eq(target.view_as(pred)).sum().item()
            test_accuracy += ta
            if idx % 10 == 0:
                print('Test Loss {}, idx {}'.format(loss.item(), idx))
        ta = 100 * test_accuracy / len(test_loader_mnist.dataset)
        tl = test_loss / len(test_loader_mnist.dataset)
        print('Test accuracy: {} Average test loss: {}'.format(ta, tl))
        return ta, tl


def shape_parameter_to_conv_net(net, params):
    """
    Shapes the parameter coming from the mean of ensembles to the network
    architecture
    """
    param_dict = dict()
    start = 0
    for key in net.state_dict().keys():
        shape = net.state_dict()[key].shape
        length = net.state_dict()[key].nelement()
        end = start + length
        param_dict[key] = params[start:end].reshape(shape)
        start = end
    return param_dict


def get_data(batch_size, device):
    """
    Provides the MNIST dataset for training and testing
    """
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    # Load data and normalize images to mean 0 and std 1
    # training set
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0], [1])
         ])
    train_loader_mnist = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=True, download=False,
                       transform=transform),
        batch_size=batch_size, shuffle=False, **kwargs)
    # test set
    test_loader_mnist = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=False, download=False,
                       transform=transform),
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader_mnist, test_loader_mnist


if __name__ == '__main__':
    with open('config.json', 'r') as jsfile:
        config = json.load(jsfile)
    path = '.'
    conv_params = {}
    files = []
    test_loss = []
    test_accuracy = []
    for file in os.listdir(path):
        if file.startswith('conv_params_'):
            files.append(file)
    files = sorted(files, key=lambda x: int(x.strip('conv_params .pt')))
    f = os.path.join(path, 'acc_loss.pt')
    if os.path.exists(f):
        losses = torch.load(f)
        test_accuracy = losses[0]
        test_loss = losses[1]
    else:
        conv_net = ConvNet()
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = config['batch_size']
        train_loader, test_loader = get_data(batch, device)
        for file in files:
            print(file)
            conv_params = torch.load(file, map_location='cpu')
            ensemble = conv_params['ensemble'].mean(0)
            ensemble = torch.from_numpy(ensemble)
            ds = shape_parameter_to_conv_net(conv_net, ensemble)
            conv_net.set_parameter(ds)
            tsa, tsl = test(conv_net, test_loader)
            test_accuracy.append(tsa)
            test_loss.append(tsl)
        torch.save((test_accuracy, test_loss), 'acc_loss.pt')
    # plot figure 1
    plot_accuracy_all_iteration_std(
        ['SGD', 'acc'], path='test_losses/')
    # plot test error for EnKF = Figure 8 a (left)
    plot_different_accuracies(range(0, 8000, 500))
    # plot test error with different acitvation functions = Figure 8b (right)
    plot_act_func_accuracies((range(0, 8000, 500)))
    # plot test errors for SGD and ADAM = Figure 5
    plot_error_sgd_adam(torch.load('test_acc.pt'))
    # plot for figure 10
    adaptive_test_loss_splitted(torch.load('acc_loss.pt'),
                                torch.load('dyn_change.pt'))
