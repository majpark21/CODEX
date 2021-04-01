import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
from load_data import DataProcesser
from results_model import top_confidence_perclass, least_correlated_set
from pattern_utils import extend_segments, create_cam, longest_segments, extract_pattern
from class_dataset import myDataset, ToTensor, RandomCrop
from dtaidistance import dtw, clustering
from models import ConvNetCam
from skimage.filters import threshold_li, threshold_mean
import os
from itertools import chain
from tqdm import tqdm
import subprocess

#%%
############################ Parameters ###################################
n_series_perclass = 125
n_pattern_perseries = 3
mode_series_selection = 'least_correlated'
thresh_confidence = 0.5  # used in least_correlated mode to choose set of series with minimal classification confidence
# patt_percmax_cam = 0.5  Replaced by Li threshold
extend_patt = 5
min_len_patt = 40
max_len_patt = 400  # length to divide by nchannel
center_patt = False
normalize_dtw = True

export_perClass = False
export_allPooled = True

assert mode_series_selection in ['top_confidence', 'least_correlated']
#%%
############################ Load model and data ##########################
data_file = 'data/ErkAkt_6GF_len240_repl2_trim100.zip'
model_file = 'forPaper/models/ERK_AKT/2019-07-04-11:21:58_ErkAkt_6GF_len240_repl2_trim100.pytorch'
selected_set = 'both'
meas_var = ['ERK', 'AKT']

batch_size = 800
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
model = torch.load(model_file)
model.eval()
model.double()
model.batch_size = batch_size
model = model.to(device)

data = DataProcesser(data_file)
data.subset(sel_groups=meas_var, start_time=0, end_time=600)
#data.dataset = data.dataset[data.dataset[data.col_class].isin([0,1])]
data.get_stats()
data.process(method='center_train', independent_groups=True)  # do here and not in loader so can use in df
data.crop_random(model.length, ignore_na_tails=True)
data.split_sets(which='dataset')
classes = tuple(data.classes.iloc[:, 1])
classes_dict = data.classes['class']
#classes = ('A', 'B')

# Random crop before to keep the same in df as the ones passed in the model
if selected_set == 'validation':
    selected_data = myDataset(dataset=data.validation_set,
                              transform=transforms.Compose([RandomCrop(output_size=model.length, ignore_na_tails=True),
                                                            ToTensor()]))
    df = data.validation_set
elif selected_set == 'training':
    selected_data = myDataset(dataset=data.train_set,
                              transform=transforms.Compose([RandomCrop(output_size=model.length, ignore_na_tails=True),
                                                            ToTensor()]))
    df = data.train_set
elif selected_set == 'both':
    try:
        selected_data = myDataset(dataset=data.dataset_cropped,
                                  transform=transforms.Compose([RandomCrop(output_size=model.length, ignore_na_tails=True),
                                                                ToTensor()]))
        df = data.dataset_cropped
    except:
        selected_data = myDataset(dataset=data.dataset,
                                  transform=transforms.Compose([RandomCrop(output_size=model.length, ignore_na_tails=True),
                                                                ToTensor()]))
        df = data.dataset

data_loader = DataLoader(dataset=selected_data,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=4)
# Dataframe used for retrieving trajectories. wide_to_long instead of melt because can melting per group of columns
df = pd.wide_to_long(df, stubnames=meas_var, i=[data.col_id, data.col_class], j='Time', sep='_', suffix='\d+')
df = df.reset_index()  # wide_to_long creates a multi-level Index, reset index to retrieve indexes in columns
df.rename(columns={data.col_id: 'ID', data.col_class: 'Class'}, inplace=True)
df['ID'] = df['ID'].astype('U32')
del data  # free memory

print('Model and data loaded.')
#%%
#################### Filter trajectories that were classified correctly with high confidence ##########################
if mode_series_selection == 'least_correlated':
    set_trajectories = least_correlated_set(model, data_loader, threshold_confidence=thresh_confidence, device=device,
                                            n=n_series_perclass, labels_classes=classes_dict)
elif mode_series_selection == 'top_confidence':
    set_trajectories = top_confidence_perclass(model, data_loader, device=device, n=n_series_perclass,
                                               labels_classes=classes_dict)

# free some memory by keeping only relevant series
selected_trajectories = set_trajectories['ID']
df = df[df['ID'].isin(selected_trajectories)]

print('Selection of trajectories done.')
#%%
#################### Extract patterns ##############################
# Store patterns in a list of np arrays for dtaidistance DTW
store_patts = {i:[] for i in classes}
model.batch_size = 1
report_filter = {'Total number of patterns': 0,
                 'Number of patterns above maximum length': 0,
                 'Number of patterns below minimum length': 0}
pbar = tqdm(total=len(selected_trajectories))
for id_trajectory in selected_trajectories:
    series_numpy = np.array(df.loc[df['ID'] == id_trajectory][meas_var]).astype('float').squeeze()
    # Row: measurement; Col: time
    if len(meas_var) >= 2:
        series_numpy = series_numpy.transpose()
    series_tensor = torch.tensor(series_numpy)
    class_trajectory = df.loc[df['ID']==id_trajectory]['Class'].iloc[0]  # repeated value through all series
    class_label = classes[class_trajectory]
    cam = create_cam(model, array_series=series_tensor, feature_layer='features',
                         device=device, clip=0, target_class=class_trajectory)
    thresh = threshold_li(cam)
    bincam = np.where(cam >= thresh, 1, 0)
    bincam_ext = extend_segments(array=bincam, max_ext=extend_patt)
    patterns = longest_segments(array=bincam_ext, k=n_pattern_perseries)
    # Filter short/long patterns
    report_filter['Total number of patterns'] += len(patterns)
    report_filter['Number of patterns above maximum length'] += len([k for k in patterns.keys() if patterns[k] > max_len_patt])
    report_filter['Number of patterns below minimum length'] += len([k for k in patterns.keys() if patterns[k] < min_len_patt])
    patterns = {k: patterns[k] for k in patterns.keys() if (patterns[k] >= min_len_patt and
                                                            patterns[k] <= max_len_patt)}
    if len(patterns) > 0:
        for pattern_position in list(patterns.keys()):
            store_patts[class_label].append(extract_pattern(series_numpy, pattern_position, NA_fill=False))
    pbar.update(1)

print(report_filter)

# Dump patterns into csv
if export_allPooled:
    concat_patts_allPooled = np.full((sum(map(len, store_patts.values())), len(meas_var) * max_len_patt), np.nan)
    irow = 0
for classe in classes:
    concat_patts = np.full((len(store_patts[classe]), len(meas_var) * max_len_patt), np.nan)
    for i, patt in enumerate(store_patts[classe]):
        if len(meas_var) == 1:
            len_patt = len(patt)
            concat_patts[i, 0:len_patt] = patt
        if len(meas_var) >= 2:
            len_patt = patt.shape[1]
            for j in range(len(meas_var)):
                offset = j*max_len_patt
                concat_patts[i, (0+offset):(len_patt+offset)] = patt[j, :]
    if len(meas_var) == 1:
        headers = ','.join([meas_var[0]+'_'+str(k) for k in range(max_len_patt)])
        fout_patt = 'output/' + meas_var +'/local_patterns/patt_uncorr_temp{}.csv.gz'.format(classe)
        if export_perClass:
            np.savetxt(fout_patt, concat_patts,
                       delimiter=',', header=headers, comments='')
    elif len(meas_var) >= 2:
        headers = ','.join([meas + '_' + str(k) for meas in meas_var for k in range(max_len_patt)])
        fout_patt = 'output/' + '_'.join(meas_var) +'/local_patterns/patt_uncorr_temp{}.csv.gz'.format(classe)
        if export_perClass:
            np.savetxt(fout_patt, concat_patts,
                       delimiter=',', header=headers, comments='')
    if export_allPooled:
        concat_patts_allPooled[irow:(irow+concat_patts.shape[0]), :] = concat_patts
        irow += concat_patts.shape[0]

if export_allPooled:
    concat_patts_allPooled = pd.DataFrame(concat_patts_allPooled)
    concat_patts_allPooled.columns = headers.split(',')
    pattID_col = [[classe] * len(store_patts[classe]) for classe in classes]
    concat_patts_allPooled['pattID'] = [j+'_'+str(i) for i,j in enumerate(list(chain.from_iterable(pattID_col)))]
    concat_patts_allPooled.set_index('pattID', inplace = True)
    fout_patt = 'output/' + '_'.join(meas_var) + '/local_patterns/patt_uncorr_temp_allPooled.csv.gz'.format(classe)
    concat_patts_allPooled.to_csv(fout_patt, header=True, index=True, compression='gzip')

print('Patterns extracted and saved.')

#%%
del (store_patts, concat_patts_allPooled, concat_patts, model,\
    df, selected_data, selected_trajectories, set_trajectories, bincam, bincam_ext, data_loader, series_tensor)
######### Build distance matrix between patterns (normalized DTW to length of series, with multivariate support) #######
if export_perClass:
    for classe in classes:
        print('Building distance matrix for class: {} \n'.format(classe))
        fout_patt = '/home/marc/Dropbox/Work/TSclass_GF/Notebooks/output/' + '_'.join(
            meas_var) + '/local_patterns/patt_uncorr_{}.csv.gz'.format(classe)
        fout_dist = '/home/marc/Dropbox/Work/TSclass_GF/Notebooks/output/' + '_'.join(
            meas_var) + '/local_patterns/uncorr_dist_norm_{}.csv.gz'.format(classe)
        if len(meas_var) == 1:
            subprocess.call(
                'Rscript --vanilla /home/marc/Dropbox/Work/TSclass_GF/dtw_multivar_distmat.R -i "{}" -o "{}" -l {} -n {} --norm {} --center {} --colid {}'.format(
                    fout_patt,
                    fout_dist,
                    max_len_patt / len(meas_var),
                    1,
                    normalize_dtw,
                    center_patt,
                    "NULL"), shell=True)
        elif len(meas_var) >= 2:
            subprocess.call(
                'Rscript --vanilla /home/marc/Dropbox/Work/TSclass_GF/dtw_multivar_distmat.R -i "{}" -o "{}" -l {} -n {} --norm {} --center {} --colid {}'.format(
                    fout_patt,
                    fout_dist,
                    max_len_patt,
                    len(meas_var),
                    normalize_dtw,
                    center_patt,
                    "NULL"), shell=True)

if export_allPooled:
    print('Building distance matrix for pooled data.')
    fout_patt = '/home/marc/Dropbox/Work/TSclass_GF/Notebooks/output/' + '_'.join(
        meas_var) + '/local_patterns/patt_uncorr_allPooled.csv.gz'
    fout_dist = '/home/marc/Dropbox/Work/TSclass_GF/Notebooks/output/' + '_'.join(
        meas_var) + '/local_patterns/uncorr_dist_norm_allPooled.csv.gz'
    if len(meas_var) == 1:
        subprocess.call(
            'Rscript --vanilla /home/marc/Dropbox/Work/TSclass_GF/dtw_multivar_distmat.R -i "{}" -o "{}" -l {} -n {} --norm {} --center {} --colid {}'.format(
                fout_patt,
                fout_dist,
                max_len_patt,
                1,
                normalize_dtw,
                center_patt,
                "pattID"), shell=True)
    elif len(meas_var) >= 2:
        subprocess.call(
            'Rscript --vanilla /home/marc/Dropbox/Work/TSclass_GF/dtw_multivar_distmat.R -i "{}" -o "{}" -l {} -n {} --norm {} --center {}  --colid {}'.format(
                fout_patt,
                fout_dist,
                max_len_patt,
                len(
                    meas_var),
                normalize_dtw,
                center_patt,
                "pattID"), shell=True)

print('Distance matrices built.')
#%%
# ################################# Cluster patterns and plot results ####################################################
# nclust = 4
# nmedoid = 3
# nseries = 16
#
# if export_perClass:
#     for classe in classes:
#         print('Cluster patterns for class: {} \n'.format(classe))
#         fout_patt = '/home/marc/Dropbox/Work/TSclass_GF/Notebooks/output/' + '_'.join(meas_var) + '/local_patterns/patt_uncorr_{}.csv.gz'.format(classe)
#         fout_dist = '/home/marc/Dropbox/Work/TSclass_GF/Notebooks/output/' + '_'.join(meas_var) + '/local_patterns/uncorr_dist_norm_{}.csv.gz.csv.gz'.format(classe)
#         fout_plot = '/home/marc/Dropbox/Work/TSclass_GF/Notebooks/output/' + '_'.join(meas_var) + '/local_patterns/uncorr_pattPlot_{}.pdf'.format(classe)
#         subprocess.call('Rscript --vanilla /home/marc/Dropbox/Work/TSclass_GF/pattern_clustering.R -d {} -p {} -o {} -l {} -n {} -c{} -m {} -t {}'.format(fout_dist,
#                                                                                                        fout_patt,
#                                                                                                        fout_plot,
#                                                                                                        max_len_patt,
#                                                                                                        len(meas_var),
#                                                                                                        nclust,
#                                                                                                        nmedoid,
#                                                                                                        nseries), shell=True)
#
# if export_allPooled:
#     fout_patt = '/home/marc/Dropbox/Work/TSclass_GF/Notebooks/output/' + '_'.join(meas_var) + '/local_patterns/patt_uncorr_allPooled.csv.gz'
#     fout_dist = '/home/marc/Dropbox/Work/TSclass_GF/Notebooks/output/' + '_'.join(meas_var) + '/local_patterns/uncorr_dist_norm_allPooled.csv.gz.csv.gz'
#     fout_plot = '/home/marc/Dropbox/Work/TSclass_GF/Notebooks/output/' + '_'.join(meas_var) + '/local_patterns/uncorr_pattPlot_allPooled.pdf'.format(classe)
#     subprocess.call('Rscript --vanilla /home/marc/Dropbox/Work/TSclass_GF/pattern_clustering.R -d {} -p {} -o {} -l {} -n {} -c{} -m {} -t {}'.format(fout_dist,
#                                                                                                fout_patt,
#                                                                                                fout_plot,
#                                                                                                max_len_patt,
#                                                                                                len(meas_var),
#                                                                                                nclust,
#                                                                                                nmedoid,
#                                                                                                nseries), shell=True)
# print('Clustering done.')
#%%
####### Alternative: Build distance matrix between patterns (only univariate case, otherwise use R dtw package) ########
#dist_matrix = {i:[] for i in classes}
#for classe in classes:
#    #dist_matrix[classe] = dtw.distance_matrix_fast(store_patts[classe])
#    dist_matrix[classe] = dtw.distance_matrix(store_patts[classe])
#    np.savetxt('output/' + meas_var + '/local_patterns/dist_{}.csv'.format(classe), dist_matrix[classe], delimiter=',')


