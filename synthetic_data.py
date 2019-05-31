#######################################
# Create synthetic time series dataset with 4 classes:
# - A: Mix of Gaussian (+) and truncated Gaussian (-); channel 2 is flat
# - B: Mix of Gaussian (-) and truncated Gaussian (+); channel 2 is flat
# - C: Equal mix of Gaussian and truncated Gaussian; channel 2 has a peak BEFORE each event in channel 1
# - D: Equal mix of Gaussian and truncated Gaussian; channel 2 has a peak AFTER each event in channel 1
#######################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

def create_channel1(length, npeak, prop_gaussian = (0.5, 1), nseries = 1, baseline = 0, sigma_noise = 0, seed = None,
                  trunc = 'left', scale_peak = 1, noise_height = (1, 1), noise_width = (0.2, 0.2)):
    """
    Create a series with a mixture of Gaussians and truncated Gaussians.
    :param nseries: int, number of series.
    :param trunc: side of the Gaussians that must be truncated, one of ['left', 'right', 'both']. 'Both' means that
    for each peak, the direction of the truncation is chosen at random.
    :param scale_peak: float, adjust height of peak (default peak height is 1).
    :param noise_height: float 2-tuple, range from which to sample individual peak height.
    :param noise_width: float 2-tuple, range from which to sample individual peak width.
    :param length: int, length of the series.
    :param npeak: int, total number of peaks (Gaussians AND truncated Gaussians) in the series
    :param prop_gaussian: float 2-tuple, range for the proportion of Gaussians.
    :param baseline: float, baseline level for the series
    :param sigma_noise: float, standard deviation of additive Gaussian noise.
    :param seed: int or None, seed for pseudo-random generators.
    :return: A numpy array.
    """
    assert prop_gaussian[0] <= prop_gaussian[1]
    assert noise_height[0] <= noise_height[1]
    assert noise_width[0] <= noise_width[1]
    assert trunc in ['left', 'right', 'both'] or trunc is None
    if sigma_noise is None:
        sigma_noise = np.mean(noise_height) / 20
    np.random.seed(seed)

    # ------------------------------------------------------------------------------------------------------------------
    # Function to create peaky trajectories
    def create_peak_traj(length, npeak, nseries, trunc, noise_height, noise_width, scale_peak):
        # Add Gaussian peaks at random position
        # Create one series per peak and sum up everything
        posPeak = np.random.randint(0, length, (nseries, npeak))
        peak_data = np.zeros((nseries, length))
        x = np.arange(0, length)
        i = 0
        for pos_vec in posPeak:
            trajGauss = []
            for pos in pos_vec:
                height_peak = np.random.uniform(noise_height[0], noise_height[1])
                sigma_peak = np.random.uniform(noise_width[0], noise_width[1])
                y = height_peak * np.exp(-((x-pos)**2)/(2*(sigma_peak**2)))
                # Use peak position to find position of maxima
                if trunc == 'both':
                    trunc = np.random.choice(['left', 'right'])
                if trunc == 'left':
                    y[:pos] = 0
                elif trunc == 'right':
                    y[pos:] = 0
                trajGauss.append(y)
            trajGauss = np.array(trajGauss)
            # Sum element-wise each peak trajectory
            peak_data[i, :] = np.sum(trajGauss, axis=0)
            i += 1
        #peak_data.clip(0.0001, out=peak_data)
        peak_data *= scale_peak
        return peak_data, posPeak

    # ------------------------------------------------------------------------------------------------------------------
    # Peak part of the signal
    # Random proportions of Gaussian from provided range
    range_nGauss = np.ceil(npeak * np.array(prop_gaussian)).astype('int')
    ltraj = []
    lpos = []
    for i in range(nseries):
        nGauss = np.random.choice(np.arange(range_nGauss[0], range_nGauss[1] + 1))
        nTrGauss = npeak - nGauss
        trajGauss, posGauss = create_peak_traj(length=length, npeak=nGauss, nseries=1, trunc=None,
                         noise_height=noise_height, noise_width=noise_width, scale_peak=scale_peak)
        trajTrGauss, posTrGauss = create_peak_traj(length=length, npeak=nTrGauss, nseries=1, trunc=trunc,
                         noise_height=noise_height, noise_width=noise_width, scale_peak=scale_peak)
        ltraj.append(trajGauss + trajTrGauss)
        # Numpy cannot concatenate empty arrays
        if posGauss.size and posTrGauss.size:
            lpos.append(np.hstack([posGauss, posTrGauss]).squeeze())
        elif posGauss.size:
            lpos.append(posGauss.squeeze())
        elif posTrGauss.size:
            lpos.append(posTrGauss.squeeze())
    # One trajectory per row
    peak_data = np.vstack(ltraj)
    peak_pos = np.vstack(lpos)

    # ------------------------------------------------------------------------------------------------------------------
    # Noise part of the signal
    # Create series of Gaussian(0, 1)
    noise_data = np.random.normal(0, sigma_noise, (nseries, length))
    noise_data += baseline

    out = np.round(peak_data + noise_data, 4)
    return out, peak_pos


def create_channel2(chan1_mat, pos_mat, shift, height = -1, baseline = 0.0):
    """
    Take positions of events of channel 1 and make a triangular signal before or after each position.
    :return:
    """
    chan2_mat = np.zeros_like(chan1_mat)
    nrow, ncol = chan1_mat.shape
    for irow in range(nrow):
        # Exclude positions out of range after shift
        pos_chan2 = pos_mat[irow, :][np.where(pos_mat[irow, :] + shift < ncol)]
        pos_chan2 = pos_chan2[np.where(pos_chan2 + shift > 0)]
        chan2_mat[irow, pos_chan2 + shift] = 1
    chan2_mat *= height
    chan2_mat += baseline
    return chan2_mat

# ----------------------------------------------------------------------------------------------------------------------
# Create synthetic dataset with the 4 classes
# Common parameters
chan1_params = {'nseries': 10000,
                'length': 750,
                'npeak': 4,
                'noise_width': (15, 25),
                'noise_height': (1, 1.5),
                'scale_peak' :1,
                'baseline': 0,
                'sigma_noise': 0,
                'trunc': 'both',
                'seed': None}

# class A: More Gaussians, flat second channel
chan1_A, pos_A = create_channel1(prop_gaussian=(0.5, 1), **chan1_params)
chan2_A = np.zeros_like(chan1_A) - 0.5

# class B: More Truncated Gaussians, flat second channel
chan1_B, pos_B = create_channel1(prop_gaussian=(0, 0.5), **chan1_params)
chan2_B = np.zeros_like(chan1_A) - 0.5

# class C: Equal mix of Gaussians and Truncated Gaussians, peak in second channel BEFORE peak in first channel
chan1_C, pos_C = create_channel1(prop_gaussian=(0.5, 0.5), **chan1_params)
chan2_C = create_channel2(chan1_mat=chan1_C, pos_mat=pos_C, shift=-30, baseline= -0.5, height=-1)

# class D: Equal mix of Gaussians and Truncated Gaussians, peak in second channel AFTER peak in first channel
chan1_D, pos_D = create_channel1(prop_gaussian=(0.5, 0.5), **chan1_params)
chan2_D = create_channel2(chan1_mat=chan1_D, pos_mat=pos_D, shift=30, baseline= -0.5, height=-1)

# Make dataframe formatted for CNN
chans1 = [chan1_A, chan1_B, chan1_C, chan1_D]
chans2 = [chan2_A, chan2_B, chan2_C, chan2_D]
df = np.hstack([np.vstack(chans1), np.vstack(chans2)])
df = pd.DataFrame(df)
df.columns = ['FRST_'+str(i) for i in range(chan1_params['length'])] + ['SCND_'+str(i) for i in range(chan1_params['length'])]
num_cols = list(df.columns)
df['class'] = np.repeat([0,1,2,3], chan1_params['nseries'])
df['ID'] = ['A_'+ str(i) for i in range(chan1_params['nseries'])] + ['B_'+ str(i) for i in range(chan1_params['nseries'])] + ['C_'+ str(i) for i in range(chan1_params['nseries'])] + ['D_'+ str(i) for i in range(chan1_params['nseries'])]
df = df[['ID', 'class'] + num_cols]
df.to_csv('data/synthetic_dataset.csv', index=False)

#-----------------------------------------------------------------------------------------------------------------------
# Split ids
ntraj = df.shape[0]
train_ids = np.random.choice(df['ID'], size = round(0.7 * ntraj), replace=False)
validation_ids = np.random.choice(np.setdiff1d(df['ID'], train_ids), size = round(0.25 * ntraj), replace=False)
test_ids = np.setdiff1d(df['ID'], train_ids)
test_ids = np.setdiff1d(test_ids, validation_ids)
vec_set = np.repeat(['train', 'validation', 'test'], [len(train_ids), len(validation_ids), len(test_ids)])

df_set = pd.DataFrame({'ID': np.concatenate([train_ids, validation_ids, test_ids]), 'set': vec_set})
df_set.to_csv('data/synthetic_id_set.csv', index=False)


# ----------------------------------------------------------------------------------------------------------------------
# DF with class IDs
df_class = pd.DataFrame({'class_ID': [0,1,2,3], 'class':['A', 'B', 'C', 'D']})
df_class.to_csv('data/synthetic_classes.csv', index=False)

# ----------------------------------------------------------------------------------------------------------------------
# Plot a sample

# ncol = 3
# nrow = 3
# for chan1, chan2, classe in zip(chans1, chans2, ['class_' + classe for classe in ['A', 'B', 'C', 'D']]):
#     for i in range(ncol*nrow):
#         plt.subplot(nrow, ncol, i+1)
#         plt.plot(np.arange(chan1_params['length']), chan1[i,:].squeeze())
#         plt.plot(np.arange(chan1_params['length']), chan2[i, :].squeeze())
#         plt.title(classe)
#     plt.show()