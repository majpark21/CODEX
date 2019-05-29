#######################################
# Create synthetic time series dataset with 4 classes:
# - A: Mix of Gaussian (+) and truncated Gaussian (-); channel 2 is flat
# - B: Mix of Gaussian (-) and truncated Gaussian (+); channel 2 is flat
# - C: Equal mix of Gaussian and truncated Gaussian; channel 2 has a peak BEFORE each event in channel 1
# - D: Equal mix of Gaussian and truncated Gaussian; channel 2 has a peak AFTER each event in channel 1
#######################################

#TODO: check probability for peak proportions

import numpy as np
import matplotlib.pyplot as plt
import datetime

def create_series(length, npeak, prop_gaussian = (0.5, 1), nseries = 1, baseline = 0, sigma_noise = 0, seed = 7,
                  trunc = 'left', scale_peak = 1, noise_height = (1, 1), noise_width = (0.2, 0.2)):
    """
    Create a series with a mixture of Gaussians and truncated Gaussians.
    :param nseries: int, number of series.
    :param trunc: side of the Gaussians that must be truncated, one of ['left', 'right'].
    :param scale_peak: float, adjust height of peak (default peak height is 1).
    :param noise_height: float 2-tuple, range from which to sample individual peak height.
    :param noise_width: float 2-tuple, range from which to sample individual peak width.
    :param length: int, length of the series.
    :param npeak: int, total number of peaks (Gaussians AND truncated Gaussians) in the series
    :param prop_gaussian: float 2-tuple, range for the proportion of Gaussians. Must either start at 0 or end at 1.
    :param baseline: float, baseline level for the series
    :param sigma_noise: float, standard deviation of additive Gaussian noise.
    :param seed: int, seed for pseudo-random generators.
    :return: A numpy array.
    """
    assert prop_gaussian[0] <= prop_gaussian[1]
    assert prop_gaussian[0] == 0 or prop_gaussian[1] == 1
    assert noise_height[0] <= noise_height[1]
    assert noise_width[0] <= noise_width[1]
    assert trunc in ['left', 'right'] or trunc is None
    if sigma_noise is None:
        sigma_noise = np.mean(noise_width) / 10
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
                noise_height_peak = np.random.uniform(noise_height[0], noise_height[1])
                sigma_peak = np.random.uniform(noise_width[0], noise_width[1])
                y = noise_height_peak * np.exp(-((x-pos)**2)/(2*(sigma_peak**2)))
                # Use peak position to find position of maxima
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
        return peak_data

    # ------------------------------------------------------------------------------------------------------------------
    # Peak part of the signal
    # Random proportions of Gaussian from provided range
    range_nGauss = np.ceil(npeak * np.array(prop_gaussian)).astype('int')
    nGauss = np.random.choice(np.arange(range_nGauss[0], range_nGauss[1] + 1))
    nTrGauss = npeak - nGauss

    trajGauss = create_peak_traj(length=length, npeak=nGauss, nseries=nseries, trunc=None,
                     noise_height=noise_height, noise_width=noise_width, scale_peak=scale_peak)
    trajTrGauss = create_peak_traj(length=length, npeak=nTrGauss, nseries=nseries, trunc=trunc,
                     noise_height=noise_height, noise_width=noise_width, scale_peak=scale_peak)
    peak_data = trajGauss + trajTrGauss

    # ------------------------------------------------------------------------------------------------------------------
    # Noise part of the signal
    # Create series of Gaussian(0, 1)
    noise_data = np.random.normal(0, sigma_noise, (nseries, length))
    noise_data += baseline

    out = np.round(peak_data + noise_data, 4)
    return out

length = 750
nseries = 6
npeak = 4
propGauss = (0.5, 1)
noise_width=(20, 20)
noise_height=(1, 1)
baseline = 5
seed = int(datetime.datetime.now().timestamp())

temp = create_series(nseries=nseries, length=length, npeak=npeak, prop_gaussian=propGauss, seed=seed, trunc="right",
                     noise_width=noise_width, noise_height=noise_height, baseline=baseline)

ncol = 3
nrow = 2
for i in range(nseries):
    plt.subplot(nrow, ncol, i+1)
    plt.plot(np.arange(length), temp[i,:].squeeze())
plt.show()

