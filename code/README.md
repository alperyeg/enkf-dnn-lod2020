# Ensemble Kalman Filter optimizing Deep Neural Networks: An alternative approach to non-performing Gradient Descent
## Code example README

## Description 
This file contains a small description on how to run the accompanying code. Supplied are five python files, a readme and a configuration file:

1. `enkf_pytorch_conv-run.py`: The main file to run the code (see [Enkf-optimization](#enkf-run)). 
1. `enkf_pytorch.py` : EnKF optimizer
1. `conv_net.py` : The Convolutional Network, is also runnable (see [gd-optimization](#gd-run)). 
1. `config.json`: Parameter configuration file
1. `README.md`: This file, description on how to run the code
1. `plot_accuracy.py`: Plots the test errors (see [plotting](#plotting)). 
1. `plot_grads_acts.py`: Plots the gradient and activation function values

*Note*: This code example focuses on the MNIST dataset run.

## How to run the code 
### Requirements
Code: 

* numpy>=1.16.0
* pytorch>=1.2.0

For plotting: 

* matplotlib>=3.0.0
* seaborn>=0.9.0

### <a id="config"></a> Configuration file
The config file `config.json` sets the parameters of the method:

* `root`: The directory where pytorch's dataloader will download and save the MNIST dataset. Default is `.` the actual folder. 
* `n_ensembles`:  Number of ensembles. Default is `5000`.
* `gamma`: Scales the identity matrix with a small scalar. Default is `0.01`
* `sigma`: Sigma of the Gaussian initialization. Default is `0.1`
* `batch_size`: The mini-batch size of the dataset Default is `64`.
* `seed`: Sets the seed for Pytorch's and Numpy's random functions. Default is `0`.
* `checkpoints`: Number of iterations until a first result is stored, starts with the first iteration. Default is `500`.
* `repetitinos`: Number of repetitions the mini-batch is presented to the network in the training phase Default is `8`.
* `epochs` : Number of epochs, only for the conv net

### <a id="enkf-run"></a> Optimization with EnKF
All python files and the configuration file should be in the same directory. `enkf_pytorch_conv-run.py` is the main python file and can be executed from the terminal with `python enkf_pytorch_conv-run.py`. It will read the parameters from the JSON configuration file `config.json`. Pytorch's dataloader will download the corresponding MNIST file into the same folder as the scripts are located if not specified otherwise (see [Configuration file](#config)). 

### <a id="gd-run"></a> Optimization with Gradient Descent
`python conv_net.py` will run the Convolutional Network with gradient descent optimization and will test different parameter settings such as different standard deviations for the initialization `stds` and two gradient descent optimizers (sgd and adam). Test results will also be stored in the actual folder.

### <a name="gd-run"></a> Creating the Plots
Two plotting scripts with corresponding data are supplied:

* `plot_accuracy.py`: Plots the test errors, i.e., Figure 1, 5, 8, 10 
  * Figure 1 needs the files ` SGD_test_accuracy_ep*.pt` and `acc_losses_ep*.pt` to be in folder in `test_losses` 
  * Figure 5 needs the file `test_acc.pt`
  * Figure 8a needs the files `acc_loss.pt`, `more_ensembles_acc_loss.pt`, `less_ensembles_acc_loss.pt`
  * Figure 8b needs the files `acc_loss.pt`, `relu_acc_loss.pt`, `tanh_acc_loss.pt`
  * Figure 10 needs the file ``
* `plot_grads_acts.py`: Plots the gradient and activation function values, i.e., Figure 3, 4, 6, 7
  * Figure 3 needs the file `act_func.npy`
  * Figure 4 needs the file `gradients.npy`
  * Figure 6 needs the files `gradients_ep*.npy`
  * Figure 7 need the files `act_func_ep*.npy`
  * Figure 9 needs the files `dyn_change.pt` and `acc_loss.pt`
  
All data files can be downloaded [here](https://mega.nz/file/vEoGFQCB#WMVfUDPRA90bEl3nmYQpnswKEeM5mEKFqI_35KiUyUs)
The contents of the folder can be merged into the code folder. 
