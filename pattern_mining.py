import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from load_data import DataProcesser
from results_model import top_classification_perclass
from pattern_utils import BinarizedCam, extend_bin, longest_strides
from class_dataset import myDataset, ToTensor, RandomCrop
from dtaidistance import dtw, clustering
from models import ConvNetCam

#%%
############################ Parameters ###################################
n_series_perclass = 75
n_pattern_perseries = 3
patt_percmax_cam = 0.5
extend_patt = 2
min_len_patt = 5
max_len_patt = 50
center_patt = True

#%%
############################ Load model and data ##########################
data_file = 'data/ErkAkt_6GF_len240.zip'
model_file = 'models/AKT/2019-05-07-10:44:38_ErkAkt_6GF_len240.pytorch'
selected_set = 'both'
meas_var = 'AKT'

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
model = torch.load(model_file)
model.eval()
model.double()
model.batch_size = 1
model = model.to(device)

data = DataProcesser(data_file)
data.subset(sel_groups=meas_var, start_time=0, end_time=600)
data.get_stats()
data.process(method='center_train', independent_groups=True)  # do here and not in loader so can use in df
data.crop_random(model.length, ignore_na_tails=True)
data.split_sets(which='dataset_cropped')
classes = tuple(data.classes.iloc[:, 1])

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
    selected_data = myDataset(dataset=data.dataset_cropped,
                              transform=transforms.Compose([#RandomCrop(output_size=model.length, ignore_na_tails=True),
                                                            ToTensor()]))
    df = data.dataset_cropped

data_loader = DataLoader(dataset=selected_data,
                         batch_size=1,
                         shuffle=True,
                         num_workers=4)
# Dataframe used for retrieving trajectories
df.columns = ['ID', 'Class'] + [i+1 for i in range(len(df.columns)-2)]
df = df.melt(['ID', 'Class'])
df.rename(columns={'variable': 'Time', 'value': 'Value'}, inplace=True)
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
    series_numpy = np.array(df.loc[df['ID'] == id_trajectory]['Value']).astype('float')
    series_tensor = torch.tensor(series_numpy)
    class_trajectory = df.loc[df['ID']==id_trajectory]['Class'].iloc[0]  # repeated value through all series
    class_label = classes[class_trajectory]
    bincam = BinarizedCam(model, array_series=series_tensor, percmax=patt_percmax_cam,
                          target_class=class_trajectory, device=device, clip=0)
    bincam_ext = extend_bin(array=bincam, max_ext=extend_patt)
    patterns = longest_strides(array=bincam_ext, k=extend_patt)
    # Filter short/long patterns
    patterns = {k: patterns[k] for k in patterns.keys() if (patterns[k] >= min_len_patt and
                                                            patterns[k] <= max_len_patt)}
    if len(patterns) > 0:
        for pattern in list(patterns.keys()):
            start, end = pattern
            store_patts[class_label].append(series_numpy[start:end])
    print('Pattern extraction for trajectory {}/{}'.format(i, len(selected_trajectories)))
    i += 1

# Center patterns beforehand because DTW is sensitive to level
if center_patt:
    for classe in classes:
        store_patts[classe] = [i - np.mean(i) for i in store_patts[classe]]

# Dump patterns into csv
for classe in classes:
    concat_patts = np.full((len(store_patts[classe]), max_len_patt), np.nan)
    for i, patt in enumerate(store_patts[classe]):
        len_patt = len(patt)
        concat_patts[i, 0:len_patt] = patt
    np.savetxt('output/' + meas_var +'/local_patterns/patt_{}.csv'.format(classe), concat_patts, delimiter=',')


#%%
#################### Build distance matrix between patterns ###########
dist_matrix = {i:[] for i in classes}
for classe in classes:
    dist_matrix[classe] = dtw.distance_matrix_fast(store_patts[classe])
    np.savetxt('output/' + meas_var + '/local_patterns/dist_{}.csv'.format(classe), dist_matrix[classe], delimiter=',')


#%%
# --> Do in R, much more flexible

#################### Clustering of patterns based on distance #########
# model1 = clustering.Hierarchical(dists_fun=dtw.distance_matrix_fast, dists_options={})
# model2 = clustering.HierarchicalTree(model1)
# cluster_idx = model2.fit(store_patts['PI3K'])
# model2.plot("myplot.png")