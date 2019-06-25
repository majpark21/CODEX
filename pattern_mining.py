import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
from load_data import DataProcesser
from results_model import top_classification_perclass
from pattern_utils import extend_segments, create_cam, longest_segments, extract_pattern
from class_dataset import myDataset, ToTensor, RandomCrop
from dtaidistance import dtw, clustering
from models import ConvNetCam
from skimage.filters import threshold_li, threshold_mean
import os

#%%
############################ Parameters ###################################
n_series_perclass = 75
n_pattern_perseries = 3
# patt_percmax_cam = 0.5  Replaced by Li threshold
extend_patt = 2
min_len_patt = 5
max_len_patt = 300
center_patt = True

#%%
############################ Load model and data ##########################
data_file = 'data/synthetic_len750.zip'
model_file = 'models/FRST_SCND/2019-06-17-15:47:03_synthetic_len750.pytorch'
selected_set = 'both'
meas_var = ['FRST', 'SCND']

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
model = torch.load(model_file)
model.eval()
model.double()
model.batch_size = 1
model = model.to(device)

data = DataProcesser(data_file)
data.subset(sel_groups=meas_var, start_time=0, end_time=750)
#data.dataset = data.dataset[data.dataset[data.col_class].isin([0,1])]
data.get_stats()
#data.process(method='center_train', independent_groups=True)  # do here and not in loader so can use in df
#data.crop_random(model.length, ignore_na_tails=True)
data.split_sets(which='dataset')
classes = tuple(data.classes.iloc[:, 1])
#classes = ('A', 'B')

# Random crop before to keep the same in df as the ones passed in the model
if selected_set == 'validation':
    selected_data = myDataset(dataset=data.validation_set,
                              transform=transforms.Compose([#RandomCrop(output_size=model.length, ignore_na_tails=True),
                                                            ToTensor()]))
    df = data.validation_set
elif selected_set == 'training':
    selected_data = myDataset(dataset=data.train_set,
                              transform=transforms.Compose([#RandomCrop(output_size=model.length, ignore_na_tails=True),
                                                            ToTensor()]))
    df = data.train_set
elif selected_set == 'both':
    try:
        selected_data = myDataset(dataset=data.dataset_cropped,
                                  transform=transforms.Compose([#RandomCrop(output_size=model.length, ignore_na_tails=True),
                                                                ToTensor()]))
        df = data.dataset_cropped
    except:
        selected_data = myDataset(dataset=data.dataset,
                                  transform=transforms.Compose([#RandomCrop(output_size=model.length, ignore_na_tails=True),
                                                                ToTensor()]))
        df = data.dataset

data_loader = DataLoader(dataset=selected_data,
                         batch_size=1,
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
top_class = top_classification_perclass(model, data_loader, classes, device, n=n_series_perclass)
# free some memory by keeping only relevant series
selected_trajectories = [i[1] for i in top_class.items()]
selected_trajectories = [i[j][1][0] for i in selected_trajectories for j in range(len(i))]
df = df[df['ID'].isin(selected_trajectories)]

print('Selection of trajectories done.')
#%%
#################### Extract patterns ##############################
# Store patterns in a list of np arrays for dtaidistance DTW
store_patts = {i:[] for i in classes}
i = 1
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
    patterns = {k: patterns[k] for k in patterns.keys() if (patterns[k] >= min_len_patt and
                                                            patterns[k] <= max_len_patt)}
    if len(patterns) > 0:
        for pattern_position in list(patterns.keys()):
            store_patts[class_label].append(extract_pattern(series_numpy, pattern_position, NA_fill=False))
    print('Pattern extraction for trajectory {}/{}'.format(i, len(selected_trajectories)))
    i += 1

# Center patterns beforehand because DTW is sensitive to level, center channels independently
if center_patt:
    for classe in classes:
        if len(store_patts[classe][0].shape) == 1:
            store_patts[classe] = [i - np.nanmean(i) for i in store_patts[classe]]
        if len(store_patts[classe][0].shape) == 2:
            store_patts[classe] = [i - np.nanmean(i, axis=1, keepdims=True) for i in store_patts[classe]]

# Dump patterns into csv
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
        fout_patt = 'output/' + meas_var +'/local_patterns/patt_full_{}.csv'.format(classe)
        np.savetxt(fout_patt, concat_patts,
                   delimiter=',', header=headers, comments='')
    elif len(meas_var) >= 2:
        headers = ','.join([meas + '_' + str(k) for meas in meas_var for k in range(max_len_patt)])
        fout_patt = 'output/' + '_'.join(meas_var) +'/local_patterns/patt_full_{}.csv'.format(classe)
        np.savetxt(fout_patt, concat_patts,
                   delimiter=',', header=headers, comments='')


#%%
################### Build distance matrix between patterns (DTW with multivariate support) #########
for classe in classes:
    print('Building distance matrix for class: {} \n'.format(classe))
    fout_patt = 'output/' + '_'.join(meas_var) + '/local_patterns/patt_full_{}.csv'.format(classe)
    fout_dist = 'output/' + '_'.join(meas_var) + '/local_patterns/dist_{}.csv'.format(classe)
    if len(meas_var) == 1:
        os.system('Rscript dtw_multivar_distmat.R -i "{}" -o "{}" -l {} -n {}'.format(fout_patt, fout_dist,
                                                                                  max_len_patt,
                                                                                  1))
    elif len(meas_var) >= 2:
        os.system('Rscript dtw_multivar_distmat.R -i "{}" -o "{}" -l {} -n {}'.format(fout_patt, fout_dist,
                                                                                  max_len_patt,
                                                                                  len(meas_var)))


# %%
################################# Cluster patterns and plot results ####################################################
nclust = 4
nmedoid = 3
nseries = 16
for classe in classes:
    print('Cluster patterns for class: {} \n'.format(classe))
    fout_patt = 'output/' + '_'.join(meas_var) + '/local_patterns/patt_full_{}.csv'.format(classe)
    fout_dist = 'output/' + '_'.join(meas_var) + '/local_patterns/dist_{}.csv'.format(classe)
    fout_plot = 'output/' + '_'.join(meas_var) + '/local_patterns/pattPlot_{}.pdf'.format(classe)
    os.system('Rscript pattern_clustering.R -d {} -p {} -o {} -l {} -n {} -c{} -m {} -t {}'.format(fout_dist,
                                                                                                   fout_patt,
                                                                                                   fout_plot,
                                                                                                   max_len_patt,
                                                                                                   len(meas_var),
                                                                                                   nclust,
                                                                                                   nmedoid,
                                                                                                   nseries))


#%%
####### Alternative: Build distance matrix between patterns (only univariate case, otherwise use R dtw package) ########
#dist_matrix = {i:[] for i in classes}
#for classe in classes:
#    #dist_matrix[classe] = dtw.distance_matrix_fast(store_patts[classe])
#    dist_matrix[classe] = dtw.distance_matrix(store_patts[classe])
#    np.savetxt('output/' + meas_var + '/local_patterns/dist_{}.csv'.format(classe), dist_matrix[classe], delimiter=',')


