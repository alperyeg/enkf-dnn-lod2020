import matplotlib.pyplot as plt
import urllib.request as request
import torch
import tarfile
import os
import numpy as np
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use('agg')


class NotMNISTLoader:
    """
    Convenience class to load the notMNIST (letters) dataset and to create
    `Pytorch` train and test `dataloaders`
    """

    def __init__(self, folder_path='.'):
        """

        :param folder_path: str, path to the notMNIST data set or folder
        """
        self.train_dataloader = None
        self.test_dataloader = None
        self.train_tragets = None
        self.test_targets = None
        self.folder_path = folder_path

    def create_dataloader(self, batch_size=32, standardize=True, test_size=0.3,
                          train_size=None, save=False, **kwargs):
        """
        Creates train and test `Pytorch` dataloaders

        :param batch_size: int, size of the batch
        :param standardize: bool, indicates if the train and test set should
                          normalized by :math:`\\frac{x - \\mu}{\\sigma}`
        :param test_size: float or int, If float: Percentage to split the
                          dataset to train and test, should be in [0,1)
                          If int: Absolute number for the test set size
        :param train_size: float or int, float or int, If float: Percentage to
                          split the dataset to train and test, should be in [0,1)
                          If int: Absolute number of the train set size
        :param save: bool, If the created dataloaders should be saved as a dictionary,
                            path filename should be given in kwargs as `filename`
                            See also `save_to_disk`
        :param kwargs: `filename`: str, indicates where to save the dataloader,
                        to be used in combination with `save`
        :return: Pytorch train and test dataloader
        """
        letters = os.listdir(self.folder_path)
        # Retrieve pictures files names
        picture_files = {}
        n_pictures = 0
        for letter in letters:
            fn = [name for name in os.listdir(os.path.join(self.folder_path, letter))
                  if name.endswith('.png')]
            picture_files[letter] = fn
        # Get the actual pictures
        data = {}
        for key in picture_files:
            files = picture_files[key]
            data[key] = []
            for f in files:
                n_pictures += 1
                try:
                    data[key].append(plt.imread(
                        os.path.join(self.folder_path, key, f)))
                except Exception as e:
                    print(f, e)

        # Merge all data to one list
        X = []
        Y = []
        X_nd = np.zeros(shape=(n_pictures, 28, 28))
        for key, list_ in data.items():
            for img in list_:
                X.append(img)
                Y.append(key)

        for i in range(len(X)):
            X_nd[i, :, :] = X[i]

        lbl_enc = LabelEncoder()
        labels = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
        lbl_enc.fit(labels)
        Y = lbl_enc.transform(Y)

        # there are inconsistencies in the length of the dataset
        # the length of the dataset is adapted to the legnth of the labels
        X_nd = X_nd[:len(Y)]

        X_train, X_test, y_train, y_test = train_test_split(X_nd, Y,
                                                            train_size=train_size,
                                                            test_size=test_size)
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        if standardize:
            mean = kwargs.get('mean', 0.5)
            std = kwargs.get('std', 0.5)
            X_train = np.divide(X_train - mean, std)
            X_test = np.divide(X_test - mean, std)
        X_train = torch.from_numpy(X_train).view(-1, 1, 28, 28)
        X_test = torch.from_numpy(X_test).view(-1, 1, 28, 28)

        train_set = TensorDataset(X_train, torch.from_numpy(y_train))
        self.train_dataloader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True)
        test_set = TensorDataset(X_test, torch.from_numpy(y_test))
        self.test_dataloader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=True)
        if save:
            self.save_to_disk(kwargs['filename'])
        return self.train_dataloader, self.test_dataloader

    def load_from_file(self, path):
        """
        Loads the dataloader dictionary from disk. The variables
        `self.train_dataloader` and `self.test_dataloader` contain
        the dictionary contents. Returns also the dictionary.

        :param path: str, path to the dictionary
        :return dict, Dictionary containing the dataloaders
        """
        dataloaders = np.load(path).item()
        self.train_dataloader = dataloaders['train_loader']
        self.test_dataloader = dataloaders['test_loader']
        return dataloaders

    def save_to_disk(self, filename='dataloader.npy'):
        """
        Saves created dataloader dictionary to disk.

        The dictionary has to keys: `train_loader` and `test_loader`

        :param filename: str, filename and path to save the dictionary
        :raise: AttributeError, If `train_loader` and `test_loader`
                are not defined
        """
        if os.path.exists(filename):
            os.remove(filename)
        if self.train_dataloader and self.test_dataloader:
            np.save(filename,
                    {'train_loader': self.train_dataloader,
                     'test_loader': self.test_dataloader})
        else:
            raise AttributeError(
                "Dataloaders missing, train_loader:{} / test_loader: {}".format(
                    self.train_dataloader, self.test_dataloader))


def _check_folder(folder_path):
    """
    Checks if the folder exists else downloads the tarball and extracts the
    contents.

    :param folder_path: str, Path to folder where the letters dataset is,
                        otherwise the file will be downloaded and extracted
    """
    if not os.path.exists(folder_path):
        tar_file = 'notmnist_large.tar.gz'
        print('Downloading tarball...')
        request.urlretrieve(
            'http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz',
            tar_file)
        print('Extracting tarball...')
        tar = tarfile.open(tar_file)
        tar.extractall()
        tar.close()


if __name__ == '__main__':
    folder_path = 'notMNIST_large'
    _check_folder(folder_path)
    print('Converting...')
    not_mnist = NotMNISTLoader(folder_path=folder_path)
    not_mnist.create_dataloader(batch_size=64, save=True, standardize=False,
                                train_size=60000, test_size=10000,
                                **{'filename': './dataloader_notmnist.npy',
                                   'shuffle': True})
