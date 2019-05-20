###################################################################################
# Determine range of LR with https://arxiv.org/abs/1506.01186; section 3.3 method #
###################################################################################

import torch
import numpy as np
from torch.utils.data import DataLoader
from load_data import DataProcesser
from torchvision import transforms
from models import ConvNetCam, ConvNetCamBi
from class_dataset import myDataset, ToTensor, Subtract, RandomShift, RandomNoise, RandomCrop, FixedCrop
from train_utils import accuracy, AverageMeter
import datetime
import time

# %% Hyperparameters
lr = [1e-5 * (10**i) for i in np.arange(0,100,0.1)]
myseed = 42
torch.manual_seed(myseed)
torch.cuda.manual_seed(myseed)
np.random.seed(myseed)

batch_size = 128
length = 200
nclass = 7
nfeatures = 20

# %% Load and process Data
data_file = 'data/ErkAkt_6GF_len240.zip'
meas_var = ['ERK', 'AKT']
data = DataProcesser(data_file)
data.subset(sel_groups=meas_var, start_time=0, end_time=600)
data.get_stats()
# data.process(method='center_train', independent_groups=True)
data.split_sets()
data_train = myDataset(dataset=data.train_set, transform=transforms.Compose([
    RandomCrop(output_size=length, ignore_na_tails=True),
    transforms.RandomApply([RandomNoise(mu=0, sigma=0.02)]),
    Subtract([data.stats['mu']['ERK']['train'], data.stats['mu']['AKT']['train']]),
    ToTensor()
]))


#%%
def EstimateLR(data_loader, lr_array, nclass=nclass):
    # Number of epochs necessary to cover all learning rates to test
    nbatch_per_epoch = len(train_loader)
    nepochs = len(lr_array) // nbatch_per_epoch
    # If batches are bigger than the number of values to test
    if nepochs == 0:
        nepochs = 1
    print('nepochs: {}'.format(nepochs))

    # ------------------------------------------------------------------------------------------------------------------
    # Model, loss, optimizer
    model = ConvNetCamBi(batch_size=batch_size, nclass=nclass, length=length, nfeatures=nfeatures)
    model.double()
    model.train()
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    
    # ------------------------------------------------------------------------------------------------------------------
    # Get adequate size of sample for nn.Conv layers
    # Add a dummy channel dimension for conv1D layer (if multivariate, treat as a 2D plane with 1 channel)
    assert len(train_loader.dataset[0]['series'].shape) == 2
    nchannel, univar_length = train_loader.dataset[0]['series'].shape
    if nchannel == 1:
        view_size = (batch_size, 1, univar_length)
    elif nchannel >= 2:
        view_size = (batch_size, 1, nchannel, univar_length)

    # ------------------------------------------------------------------------------------------------------------------
    # Run through batches and increase LR between each batch
    offset_idx = 0
    loss_out = []
    for epoch in range(nepochs):
        for i_batch, sample_batch in enumerate(train_loader):
            if i_batch > len(lr_array)-1:
                break
            print('Batch number: {}/{}; LR: {}'.format(i_batch + offset_idx, len(lr_array), lr_array[i_batch + offset_idx]))

            optimizer = torch.optim.Adam(model.parameters(), lr=lr_array[i_batch + offset_idx], betas=(0.9, 0.999))
            series, label = sample_batch['series'], sample_batch['label']
            if cuda_available:
                series, label = series.cuda(), label.cuda()
            series = series.view(view_size)
            prediction = model(series)

            # Backprop and update
            loss_before = criterion(prediction, label)
            optimizer.zero_grad()
            loss_before.backward()
            optimizer.step()
            
            # Record loss after update for this batch and LR
            loss_after = criterion(prediction, label)
            loss_out.append(loss_after.cpu().detach().numpy())
        offset_idx += nbatch_per_epoch
    return loss_out


# Define train and loader
# Dataloaders
train_loader = DataLoader(dataset=data_train,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4,
                          drop_last=True)

t0 = time.time()
losses = EstimateLR(data_loader=train_loader, lr_array=lr, nclass=nclass)
t1 = time.time()

print('Elapsed time: {}'.format(t1 - t0))
print(losses)
