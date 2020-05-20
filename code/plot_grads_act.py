import collections
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch


sns.set(style="white")
sns.set_color_codes("dark")
sns.set_context("paper", font_scale=1.5, rc={
                "lines.linewidth": 2., "grid.linewidth": 0.1})


def activation_functions_dist_iteration(act_func, savepath=''):
    """
    Distribution of the activation values per iteration
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True)
    ax1.set_title('Layer 1')
    ax2.set_title('Layer 2')
    ax3.set_title('Layer 3')

    act1 = act_func.get('act1')
    act2 = act_func.get('act2')
    act3 = act_func.get('act3')
    [sns.distplot(a.mean(0).ravel(), ax=ax1,
                  label='iteration {}'.format(i * 200))
     for i, a in enumerate(act1)]
    [sns.distplot(a.mean(0).ravel(), ax=ax2,
                  label='iteration {}'.format(i * 200)) for i, a in
     enumerate(act2)]
    [sns.distplot(a.mean(0).ravel(), ax=ax3,
                  label='iteration {}'.format(i * 200)) for i, a in
     enumerate(act3)]
    plt.legend()
    plt.savefig(savepath)
    plt.show()


def activation_functions_dist_layer(act_func, iteration=-1, savepath=''):
    act1 = act_func.get('act1')
    act2 = act_func.get('act2')
    act3 = act_func.get('act3')
    sns.distplot(act1[iteration][0].ravel(), label='Layer 1')
    sns.distplot(act2[iteration][0].ravel(), label='Layer 2')
    sns.distplot(act3[iteration][0].ravel(), label='Layer 3')
    plt.title('Activation Value at Iteration {}'.format(iteration))
    plt.legend()
    plt.savefig(savepath)
    plt.show()


def activation_functions_mean_std(act_func, errorevery=10, savepath=''):
    act1_mean = np.array(act_func.get('act1_mean'))
    act2_mean = np.array(act_func.get('act2_mean'))
    act3_mean = np.array(act_func.get('act3_mean'))
    act1_std = np.array(act_func.get('act1_std'))
    act2_std = np.array(act_func.get('act2_std'))
    act3_std = np.array(act_func.get('act3_std'))

    plt.errorbar(range(len(act1_mean)), act1_mean, act1_std,
                 errorevery=errorevery, alpha=0.8, label='layer 1')
    plt.errorbar(range(len(act2_mean)), act2_mean, act2_std,
                 errorevery=errorevery, alpha=0.5, label='layer 2')
    plt.errorbar(range(len(act3_mean)), act3_mean, act3_std,
                 errorevery=errorevery, alpha=0.4, label='layer 3')
    plt.xlabel('Iterations of mini-batches')
    plt.ylabel('Activation value')
    plt.legend()
    plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def gradients_dist_layer(grads, opt, savepath=''):
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, sharex=False if opt == 'SGD' else True)
    ax1.set_title('Layer 1')
    ax2.set_title('Layer 2')
    ax3.set_title('Layer 3')
    ax2.set_xlabel('Backpropagated Gradients')
    ax1.set_ylabel('Counts')

    grad1 = grads.get('conv1_grad')
    grad2 = grads.get('conv2_grad')
    grad3 = grads.get('fc1_grad')
    [sns.distplot(a.mean(0).ravel(), ax=ax1,
                  label='iteration {}'.format(i * 200))
     for i, a in enumerate(grad1)]
    [sns.distplot(a.mean(0).ravel(), ax=ax2,
                  label='iteration {}'.format(i * 200)) for i, a in
     enumerate(grad2)]
    [sns.distplot(a.mean(0).ravel(), ax=ax3,
                  label='iteration {}'.format(i * 200)) for i, a in
     enumerate(grad3)]
    if opt == 'Adam':
        ax2.set_ylim(0, 1000)
        ax3.set_ylim(0, 0.5e9)
    else:
        for label in ax1.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        for label in ax2.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
    plt.legend()
    plt.show()
    fig.savefig(savepath, bbox_inches='tight', pad_inches=0.1)


def gradients_mean_std(grads, errorevery=10, savepath=''):
    grad1_mean = grads['conv1_grad_mean']
    grad2_mean = grads['conv2_grad_mean']
    grad3_mean = grads['fc1_grad_mean']

    grad1_std = grads['conv1_grad_std']
    grad2_std = grads['conv2_grad_std']
    grad3_std = grads['fc1_grad_std']

    plt.errorbar(range(len(grad1_mean)), grad1_mean, grad1_std,
                 errorevery=errorevery, alpha=0.8, label='layer 1')
    plt.errorbar(range(len(grad2_mean)), grad2_mean, grad2_std,
                 errorevery=errorevery, alpha=0.5, label='layer 2')
    plt.errorbar(range(len(grad3_mean)), grad3_mean, grad3_std,
                 errorevery=errorevery, alpha=0.5, label='layer 3')
    plt.legend()
    plt.savefig(savepath)
    plt.show()


def gradients_per_epoch(grads, errorevery=10, savepath=''):
    grad1_mean = []
    grad2_mean = []
    grad3_mean = []
    grad1_std = []
    grad2_std = []
    grad3_std = []
    for k, v in grads.items():
        grad1_mean += v['conv1_grad_mean']
        grad2_mean += v['conv2_grad_mean']
        grad3_mean += v['fc1_grad_mean']
        grad1_std += v['conv1_grad_std']
        grad2_std += v['conv2_grad_std']
        grad3_std += v['fc1_grad_std']
    plt.errorbar(range(len(grad1_mean)), grad1_mean, grad1_std,
                 errorevery=errorevery, alpha=0.8, label='layer 1')
    plt.errorbar(range(len(grad2_mean)), grad2_mean, grad2_std,
                 errorevery=errorevery, alpha=0.5, label='layer 2')
    plt.errorbar(range(len(grad3_mean)), grad3_mean, grad3_std,
                 errorevery=errorevery, alpha=0.5, label='layer 3')
    plt.xlabel('Epochs')
    plt.ylabel('Gradients')
    plt.xticks(range(0, 5000, 1000), range(0, 50, 10))
    plt.legend()
    plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def activation_dist_per_epoch(acts, savepath=''):
    act1_mean = []
    act2_mean = []
    act3_mean = []
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=False)
    for i, (k, v) in enumerate(acts.items()):
        act1_mean = np.array(v['act1_mean'])
        act2_mean = np.array(v['act2_mean'])
        act3_mean = np.array(v['act3_mean'])
        sns.distplot(act1_mean, label='Epoch {}'.format(i * 10), ax=ax1)
        sns.distplot(act2_mean, label='Epoch {}'.format(i * 10), ax=ax2)
        sns.distplot(act3_mean, label='Epoch {}'.format(i * 10), ax=ax3)
    ax1.set_ylim(0, 150)
    ax2.set_ylim(0, 100)
    ax1.set_title('Layer 1')
    ax2.set_title('Layer 2')
    ax3.set_title('Layer 3')
    ax2.set_xlabel('Activation values')
    ax1.set_ylabel('Counts')
    plt.legend()
    plt.show()
    fig.savefig(savepath, bbox_inches='tight', pad_inches=0.1)


def load_epochs_grad_act(path, startswith):
    d = collections.OrderedDict()
    files = []
    for file in os.listdir(path):
        if file.startswith(startswith):
            files.append(file)
    files.sort()
    for file in files:
        d[file] = np.load(os.path.join(path, file), allow_pickle=True).item()
    return d


def activations_mean_std_error(act_func, errorevery=10, savepath=''):
    act1_mean = act_func['act1_mean'][::8]
    act1_std = act_func['act1_std'][::8]
    act2_mean = act_func['act2_mean'][::8]
    act2_std = act_func['act2_std'][::8]
    act3_mean = act_func['act3_mean'][::8]
    act3_std = act_func['act3_std'][::8]

    plt.errorbar(range(len(act1_mean)), act1_mean, act1_std,
                 errorevery=errorevery, alpha=0.8, label='layer 1')
    plt.errorbar(range(len(act2_mean)), act2_mean, act2_std,
                 errorevery=errorevery, alpha=0.5, label='layer 2')
    plt.errorbar(range(len(act3_mean)), act3_mean, act3_std,
                 errorevery=errorevery, alpha=0.6, label='layer 3')
    plt.xlabel('Iteration of mini-batches')
    plt.ylabel('Activation value')
    plt.legend()
    plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def weight_distributions_per_layer(iterations,
                                   suptitle='Ensembles - Weights',
                                   bins=None,
                                   savepath='',
                                   rand=False):
    fig, axes = plt.subplots(1, len(iterations), sharey=True)
    # fig.tight_layout()
    if rand:
        rnd = np.random.randint(0, len(iterations[0][1]['ensemble']))
    for i, (key, params) in enumerate(iterations.items()):
        if rand:
            dist = np.array(params['ensemble'])[rnd],
        else:
            dist = np.array(params['ensemble']).mean(0),
        ax = axes[i]
        sns.distplot(dist, bins=bins, color='b', ax=ax)
        ax.set_title('{}'.format(
            key.strip('conv_params_ .pt')))
    if rand:
        fig.suptitle(suptitle + '\n' + 'Ensemble {}'.format(rnd))
    else:
        fig.suptitle(suptitle)
    for a in axes:
        tkl = a.xaxis.get_ticklabels()
        [label.set_visible(False) for label in tkl[::2]]

    axes[0].set_ylabel('Counts')
    axes[1].set_xlabel('Ensemble Distribution')
    plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def load_iter_act(path, startswith):
    d = collections.OrderedDict()
    files = []
    for file in os.listdir(path):
        if file.startswith(startswith):
            files.append(file)

    files = sorted(files, key=lambda x: int(x.strip('conv_params .pt')))
    for file in files:
        d[file] = torch.load(file)
    return d


if __name__ == '__main__':
    path = "."
    # Plots gradient and activation function related figures for SGD and Adam
    for opt in ['SGD', 'Adam']:
        # Figure 3a & 6a
        act_funct = np.load(os.path.join(
            path, opt, 'act_func.npy'), allow_pickle=True).item()
        activation_functions_mean_std(act_funct, savepath=os.path.join(
            path, '{}_activation_mean_std.pdf'.format(opt)))

        # Figure 3b & 6b
        gradients = np.load(os.path.join(
            path, opt, 'gradients.npy'), allow_pickle=True).item()
        gradients_dist_layer(gradients, opt, savepath=os.path.join(
            path, '{}_grad_distribution_per_layer.pdf'.format(opt)))

        # Figure 4a & 7a
        gradients = load_epochs_grad_act(os.path.join(
            path, opt), 'gradients_ep')
        gradients_per_epoch(
            gradients, savepath='{}_grads_per_epoch.pdf'.format(opt))

        # Figure 4b & 7b
        acts = load_epochs_grad_act(os.path.join(
            path, opt), 'act_func_ep')
        activation_dist_per_epoch(
            acts, savepath='{}_activation_mean_std_per_epoch.pdf'.format(opt))

    # EnKF plots for activation functions
    # Figure 9
    activations = load_iter_act('.', startswith='conv_params_')
    weight_distributions_per_layer(activations,
                                   savepath='enkf_dist_ensembles_iterations.pdf',
                                   rand=False, suptitle='')
    # Figure 10
    activations_mean_std_error(torch.load('conv_params.pt')['act_func'],
                               savepath='enkf_act_func_mean_std.pdf')
