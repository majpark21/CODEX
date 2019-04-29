###################################################################
# Define dataset classes and preprocessing for Pytorch DataLoader #
###################################################################

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from copy import deepcopy


#TODO: make a single dataset class for both bi and univariate
class myDataset(Dataset):
    """Standard dataset object with ID, class"""

    def __init__(self, dataset=None, csv_file=None, transform=None):
        """
        First column: series ID
        Second column: series class label
        Other columns: measurement
        Args:
            data (pandas object):
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if dataset is not None and csv_file:
            raise ValueError('Only one of data and csv_file can be provided.')
        if dataset is not None:
            self.dataset = dataset
        elif csv_file:
            self.dataset = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        identifier = self.dataset.iloc[idx, 0]
        label = np.array(self.dataset.iloc[idx, 1])
        label = label.astype('int')
        series = self.dataset.iloc[idx, 2:].as_matrix()
        series = series.astype('float')
        sample = {'identifier': identifier, 'series': series, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


# Todo: add choice of image or channel mode for shape series
class myDatasetBi(Dataset):
    """Paolo dataset bivariate"""

    def __init__(self, csv_file, length, transform=None):
        """
        First column: series ID
        Second column: series class label
        Other columns: measurements, concatenated with equal length
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            length (integer): Length of each univariate series.
        """
        self.dataset = pd.read_csv(csv_file, header=None)
        self.transform = transform
        self.length = length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        identifier = self.dataset.iloc[idx, 0]
        label = np.array(self.dataset.iloc[idx, 1])
        label = label.astype('int')
        # Bivariate, create 2 channels. 1st col ID, 2nd col: label
        series = np.vstack((self.dataset.iloc[idx, 2:(2+self.length)].as_matrix(),
                            self.dataset.iloc[idx, (2 + self.length):].as_matrix()))
        series = series.astype('float')
        sample = {'identifier': identifier, 'series': series, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


# ======================================================================================================================
# Transformation and augmentation

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        series, label, identifier = sample['series'], sample['label'], sample['identifier']
        return {'series': torch.from_numpy(series),
                'label': torch.from_numpy(label),
                'identifier': identifier}


class Subtract(object):
    """Subtract fixed value from series. Useful to remove mean of training set.

    Args:
        value (float, int): value to subtract
    """

    def __init__(self, value):
        assert isinstance(value, (int, float))
        self.value = value

    def __call__(self, sample):
        series, label, identifier = sample['series'], sample['label'], sample['identifier']
        value_array = np.ones_like(series) * self.value
        series -= value_array
        return {'series': series,
                'label': label,
                'identifier': identifier}


class RandomNoise(object):
    """Add Gaussian noise.

    Args:
        mu (int, float): Mean of the noise
        sigma (int, float): Standard deviation of the noise
    """

    def __init__(self, mu, sigma):
        assert isinstance(mu, (int, float))
        assert isinstance(sigma, (int, float))
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        series, label, identifier = sample['series'], sample['label'], sample['identifier']
        noise = np.random.normal(self.mu, self.sigma, series.shape)
        series += noise
        return {'series': series,
                'label': label,
                'identifier': identifier}


class RandomShift(object):
    """Shift series left or right and pad with background.

    Args:
        max_shift (int): maximum shift, shift uniformly picked in [0, max_shift]
        direction (str): shift to the 'left', 'right' or 'random'
        method (float): method used for padding. Can be any valid argument for numpy.pad(mode=) other than function.
    """

    def __init__(self, max_shift, direction='random', pad='edge', **kwargs):
        assert isinstance(max_shift, int)
        assert isinstance(direction, str)
        assert isinstance(pad, str)
        assert direction in ['left', 'right', 'random']
        assert pad in ['constant', 'edge', 'linear_ramp', 'maximum', 'mean',
                       'median', 'minimum', 'reflect', 'symmetric', 'wrap']
        self.max_shift = max_shift
        self.direction = direction
        self.pad = pad
        self.extra_args = kwargs

    def __call__(self, sample):
        series, label, identifier = sample['series'], sample['label'], sample['identifier']
        shift_width = np.random.randint(0, self.max_shift+1)

        # Pad on both sides then trim
        if(shift_width > 0):
            direction = deepcopy(self.direction)
            if direction == 'random':
                direction = np.random.choice(['left', 'right'], 1)[0]
            series = np.pad(series, pad_width=shift_width, mode=self.pad, **self.extra_args)
            # Trim 2*shift_width because na.pad on both sides
            if direction == 'left':
                series = series[0:(len(series)-2*shift_width)]
            elif direction == 'right':
                series = series[2*shift_width:]  # becomes -1 if shift is 0

        return {'series': series,
                'label': label,
                'identifier': identifier}


# TODO: adapt to multivariate, check website pytorch tutorial
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (int): Desired output size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, sample):
        series, label, identifier = sample['series'], sample['label'], sample['identifier']
        length = len(series)
        new_length = self.output_size

        left = np.random.randint(0, length - new_length)
        series = series[left: left + new_length]

        return {'series': series,
                'label': label,
                'identifier': identifier}


# TODO: adapt to multivariate, check website pytorch tutorial
class FixedCrop(object):
    """Crop a series between fixed indexes.

    Args:
        start (int): Start index
        end (int): End index (exclusive)
    """

    def __init__(self, start, end):
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert end > start
        self.start = start
        self.end = end

    def __call__(self, sample):
        series, label, identifier = sample['series'], sample['label'], sample['identifier']
        series = series[self.start:self.end]

        return {'series': series,
                'label': label,
                'identifier': identifier}
