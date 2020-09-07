from sklearn.manifold import TSNE
import numpy as np
import random
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from class_dataset import myDataset, ToTensor, RandomCrop, Subtract
from load_data import DataProcesser
from torchvision import transforms
import re
from models import ConvNetCamBi
from copy import copy
from utils import model_output_app, frange
from plot_maps import create_cam, create_gbackprop, select_series
from copy import deepcopy
from skimage.filters import threshold_li, threshold_mean
from tkinter import filedialog, Tk
from functools import reduce
import argparse

#Todo: ID upload, selection hover/click mode, correct centering and dashed lines when time is not increasing 1 by 1
def parseArguments_overlay():
    parser = argparse.ArgumentParser(description='Project the CNN features with tSNE and browse interactively.')
    parser.add_argument('-m', '--model', help='str, path to the model file', type=str)
    parser.add_argument('-d', '--data', help='str, path to the model file', type=str)
    parser.add_argument('-s', '--set', help='str, set to project. Must be one of ["all", "train", "validation", "test"]. Default to "all".',
                        type=str, default='all')
    parser.add_argument('-b', '--batch', help='int, size of the batch when passing data to the model. '
                                              'Increase as high as GPU memory allows for speed up.'
                                              ' Must be smaller than the number of series selected,'
                                              ' Default is set to 1/10 of the dataset size.',
                        type=int, default=-1)
    parser.add_argument('--measurement', help='list of str, names of the measurement variables. In DataProcesser convention,'
                                              ' this is the prefix in a column name that contains a measurement'
                                              ' (time being the suffix). Pay attention to the order since this is'
                                              ' how the dimensions of a sample of data will be ordered (i.e. 1st in'
                                              ' the list will form 1st row of measurements in the sample,'
                                              ' 2nd is the 2nd, etc...). Leave empty for automatic detection.',
                        type=str, default='', nargs='*')
    parser.add_argument('--seed', help='int, seed for random, ensures reproducibility. Default to 7.',
                        type=int, default=7)
    parser.add_argument('--start', help='int, start time range for selecting data. Useful to ignore part of the '
                                        'data were irrelevant measurement were acquired. Set to -1 for automatic detection.',
                        type=int, default=-1)
    parser.add_argument('--end', help='int, end time range for selecting data. Useful to ignore part of the '
                                        'data were irrelevant measurement were acquired. Set to -1 for automatic detection.',
                        type=int, default=-1)
    args = parser.parse_args()
    return(args)

args = parseArguments_overlay()
# ----------------------------------------------------------------------------------------------------------------------
# Inputs
myseed = args.seed
np.random.seed(myseed); random.seed(myseed); torch.manual_seed(myseed)
# Parameters
data_file = args.data

start_time = None if args.start==-1 else args.start
end_time = None if args.end==-1 else args.end
measurement = None if args.measurement=='' else args.measurement
selected_classes = None
perc_selected_ids = 1  # Select only percentile of all trajectories, not always useful to project them all and slow
batch_sz = args.batch  # set as high as GPU memory can handle for speed up

rand_crop = True
set_to_project = args.set  # one of ['all', 'train', 'validation', 'test']

# ----------------------------------------------------------------------------------------------------------------------
# Model Loading
model_file = args.model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = torch.load(model_file) if torch.cuda.is_available() else torch.load(model_file, map_location='cpu')
net.eval()
net.double()
net = net.to(device)
length = net.length

# ----------------------------------------------------------------------------------------------------------------------
# Data Loading, Subsetting, Preprocessing
data = DataProcesser(data_file)
measurement = data.detect_groups_times()['groups'] if measurement is None else measurement
start_time = data.detect_groups_times()['times'][0] if start_time is None else start_time
end_time = data.detect_groups_times()['times'][1] if end_time is None else end_time
data.subset(sel_groups=measurement, start_time=start_time, end_time=end_time)
data.get_stats()
if selected_classes is not None:
    data.dataset = data.dataset[data.dataset[data.col_class].isin(selected_classes)]
data.split_sets(which='dataset')

print('Start time: {}; End time: {}; Measurement: {}'.format(start_time, end_time, measurement))

# ----------------------------------------------------------------------------------------------------------------------
# df used to plot measurements
assert set_to_project in ['all', 'train', 'validation', 'test']
if set_to_project == 'all':
    df = deepcopy(data.dataset)
elif set_to_project == 'train':
    df = deepcopy(data.train_set)
elif set_to_project == 'validation':
    df = deepcopy(data.validation_set)
elif set_to_project == 'test':
    df = deepcopy(data.test_set)
print('Size of raw dataframe: {}'.format(df.shape))
df.rename(columns={data.col_id: 'ID', data.col_class: 'Class'}, inplace = True)
selected_ids = np.random.choice(df.loc[:,'ID'].unique(), round(perc_selected_ids * df.shape[0]), replace=False)
df = df[df['ID'].isin(selected_ids)]
print('Size of selected dataframe: {}'.format(df.shape))

if batch_sz == -1:
    batch_sz = round(df.shape[0]/10)
print('Batch size: {}'.format(batch_sz))
assert batch_sz <= df.shape[0]
# Split for each measurement, melt and append.
ldf = []
for meas in measurement:
    col_meas = [i for i in df.columns if re.match('^{}_'.format(meas), i)]
    temp = df[['ID', 'Class'] + col_meas].melt(['ID', 'Class'])
    temp['Time'] = temp['variable'].str.extract('([0-9]+$)')
    temp['variable'] = temp['variable'].str.replace('_[0-9]+$', '')
    ldf.append(temp)
df = pd.concat(ldf)
del temp, ldf
df.sort_values(['ID', 'Time'])
max_slider = df['value'].max()
min_slider = df['value'].min()
default_slider = df['value'].quantile([0.005, 0.995])

# ----------------------------------------------------------------------------------------------------------------------
# Prepare dataloader for t-SNE, set high batch_size to speed up
subtract_numbers = [data.stats['mu'][meas]['train'] for meas in measurement]
if rand_crop:
    ls_transforms = transforms.Compose([
        Subtract(subtract_numbers),
        RandomCrop(output_size=length, ignore_na_tails=True, export_crop_pos=True),
        ToTensor()])
else:
    ls_transforms = transforms.Compose([Subtract(subtract_numbers),
                                        ToTensor()])

mydataset = myDataset(dataset=data.dataset[data.dataset['ID'].isin(selected_ids)], transform=ls_transforms)
mydataloader = DataLoader(dataset=mydataset,
                          batch_size=batch_sz,
                          shuffle=False,
                          drop_last=False)

# ----------------------------------------------------------------------------------------------------------------------
# Classes object definition
classes = tuple(data.classes[data.col_classname])
classes_col = data.classes[data.col_classname]
if selected_classes is not None:
    classes = [i for i in classes if i
               in list(data.classes[data.classes[data.col_class].isin(selected_classes)][data.col_classname])]
    classes = tuple(classes)
    classes_dict = data.classes[data.classes[data.col_class].isin(selected_classes)].to_dict()[data.col_classname]
else:
    classes_dict = data.classes.to_dict()[data.col_classname]

net.batch_size = batch_sz  # Learn representations over the whole dataset at once if equal to dataset length
# ----------------------------------------------------------------------------------------------------------------------
# Copy with unit batch for creating CAM and backprop
net_unit_batch = deepcopy(net)
net_unit_batch.batch_size = 1
# Redefine without random crop for CAM plotting
mydataset_unit_batch = myDataset(dataset=data.dataset[data.dataset[data.col_id].isin(selected_ids)],
                                 transform=transforms.Compose([ToTensor()]))
mydataloader_unit_batch = DataLoader(dataset=mydataset_unit_batch,
                          batch_size=1,
                          shuffle=False,
                          drop_last=False)
del data  # Free some memory

# ----------------------------------------------------------------------------------------------------------------------
# Compute tSNE coords, model output, store cropping info for plotting
# No grad to save GPU memory, no backward pass is done here
@torch.no_grad()
def tsne(model, dataloader, device, layer_feature='pool', ncomp=2, ini='pca', perplex=30.0, lr=200.0, niter=250):
    df_out = model_output_app(model, dataloader, export_prob=True, export_feat=True, device=device,
                              feature_layer=layer_feature, export_crop_pos=rand_crop)
    df_out['Class'].replace(classes_col, inplace=True)
    prob_cols = [i for i in df_out.columns if i.startswith('Prob_')]
    feat_cols = [i for i in df_out.columns if i.startswith('Feat_')]

    label = np.array(df_out['Class'])
    identifier = np.array(df_out['ID'])

    # Df with classification output, used to build html table when hovering points and export of selection
    global DF_PROBS
    DF_PROBS = deepcopy(df_out[prob_cols])  # copy avoids pandas warning about modifying copy
    DF_PROBS.index = df_out['ID']
    old_labels = {col: re.search('\d+$', col) for col in DF_PROBS.columns.values}
    new_labels = {col: re.sub('\d+$', classes_col[int(old_labels[col].group())], col)
                  for col in old_labels if old_labels[col]}
    DF_PROBS.rename(new_labels, axis='columns', inplace=True)
    DF_PROBS.columns = [re.sub('Prob_', '', col) for col in DF_PROBS.columns]

    # Df with CNN features, used to export of selection
    global DF_FEATS
    DF_FEATS = df_out[feat_cols]
    DF_FEATS.index = df_out['ID']

    # Df with positions of crop, used to plot CAM maps at right positions, highlight part seen by model
    if rand_crop:
        global DF_CROP
        DF_CROP = df_out[['ID', 'Crop_start', 'Crop_end']]

    # Reshape in a single big array and compute TSNE coord
    # One row for each trajectory, columns according to the number of features
    feature_blobs_array = np.array(df_out[feat_cols])
    feature_embedded = TSNE(n_components=ncomp, init=ini, perplexity=perplex,
                            learning_rate=lr, n_iter=niter, verbose=2).fit_transform(feature_blobs_array)

    return feature_embedded, label, identifier


# ----------------------------------------------------------------------------------------------------------------------
# App layout
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=['Visualize features learnt by selected layer with t-SNE.',
#==== Parameters division ====
html.Div([
    # 1st column --------------
    html.Div([html.Div(['Layer to project:', dcc.Dropdown(id='drop-layer', value='pool', clearable=False,
                           options=[{'label': i, 'value': i} for i in list(net._modules.keys())])]),
              html.Div(['Initialization:', dcc.Dropdown(id='drop-init', value='pca', clearable=False,
                           options=[{'label': i, 'value': i} for i in ['random', 'pca']])]),
              html.Div(['N dimensions:', dcc.Dropdown(id='drop-ndim', value=2, clearable=False,
                           options=[{'label':i, 'value':j} for i,j in [('2D', 2), ('3D', 3)]])])],
    style={'width': '15%', 'display': 'inline-block', 'padding-left': '0.5%', 'vertical-align': 'bottom'}),

    # 2nd column ---------------
    html.Div([html.Div(['Learning rate:', dcc.Input(id='input-lr', value=600, type='number',
                        placeholder='Learning Rate (default = 200)', min=1)]),
              html.Div(['Perplexity: ', dcc.Input(id='input-perp', value=50, type='number',
                        placeholder='Perplexity (default = 30)', min=1)]),
              html.Div(['N iterations: ', dcc.Input(id='input-niter', value=250, type='number',
                        placeholder='Max number of iteration', min=250)])
             ],
    style={'width': '15%', 'display': 'inline-block', 'padding-left': '5%', 'vertical-align': 'bottom'}),

    # 3rd column --------------
    html.Div([
          html.Div(['Color overlay for trajectories:', dcc.Dropdown(id='drop-overlay', value='None', clearable=False,
                     options=[{'label': i, 'value': i} for i in ['None', 'CAM', 'Guided Backprop']])]),
          html.Div(['Target class overlay:', dcc.Dropdown(id='drop-target', value='prediction', clearable=False, placeholder='Select overlay first',
                                                                  options=[{'label':'Prediction', 'value': 'prediction'}] +
                                                                          [{'label': v, 'value': k} for k,v in classes_dict.items()])],
                   id='div-target'),
          html.Div(['Binarization overlay:', dcc.Dropdown(id='drop-bin', value='No bin', clearable=False, placeholder='Select overlay first',
                                                          options=[{'label': i, 'value': i} for i in ['No bin',
                                                                                                      'Li Threshold',
                                                                                                      'Mean Threshold']])],
                   id='div-bin')
          ],
    style={'width': '15%', 'display': 'inline-block', 'padding-left': '5%', 'vertical-align': 'bottom'}),

    # 4th column ----------------
    html.Div([html.Div(id='div-xaxis', style={} if rand_crop else {'display': 'none'}, children=[
              html.Div([dcc.Checklist(id='check-xrange', value=[], options=[{'label': ' Center x-axis on network input',
                                                                              'value': True}])],
                       style={'padding-bottom': '0%', 'display': 'inline-block'}),
              html.Div([dcc.Checklist(id='check-xshow', value=[True], options=[{'label': ' Show borders of network input',
                                                                                  'value': True}])],
                       style={'padding-bottom': '2%', 'display': 'inline-block'})]),
              html.Div(['Range of y-axis:',
              dcc.RangeSlider(id='slider-yrange',
                              min=min_slider,
                              max=max_slider,
                              step=(max_slider-min_slider)/100,
                              value=default_slider,
                              allowCross=False,
                              updatemode='drag',
                              marks={i: {'label': str(round(i, 2))} for i in
                                     frange(min_slider, max_slider - 1e-9 + (max_slider - min_slider) / 5, (max_slider - min_slider) / 5)}
                              )], style={'padding-bottom':'10%'}),
              html.Button(id='submit-tsne', n_clicks=0, children='Run t-SNE', style={'background-color': '#008CBA',
                                                                        'color': 'white'})],
    style = {'width': '20%', 'display': 'inline-block', 'padding-left': '5%', 'vertical-align': 'bottom'}),

],style={'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px 30px',
         'display': 'block'
}),
#==== Plotting division ====
#---- tSNE scatterplot #----
html.Div([
    dcc.Graph(id='plot-tsne'),
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
#---- Time-series plotting ----
html.Div([
     dcc.Graph(id='plot-meas'),
  ], style={'display': 'inline-block', 'width': '49%'}),
#---- Table probability and export button ----
html.Div([
  html.Button(id='button-export', n_clicks=0, children='Export selection', style={'display': 'inline-block',
                                                                                  'background-color': '#008CBA',
                                                                                  'color':'white',
                                                                                  'float':'left'})],
    style={'padding-left': '5%'}
),
html.Div(id='table-proba', style={'display': 'inline-block',
                                      'float': 'right'})
])


# ----------------------------------------------------------------------------------------------------------------------
# Disable dropdowns of CAMs/backprop if no overlay selected
@app.callback(
    [Output('drop-target', 'disabled'), Output('drop-bin', 'disabled')],
    [Input('drop-overlay', 'value')]
)
def disable_drops(overlay_value):
    if overlay_value == 'None':
        return True, True
    else:
        return False, False

@app.callback(
    [Output('div-target', 'style'), Output('div-bin', 'style')],
    [Input('drop-overlay', 'value')],
)
def grey_out(overlay_value):
    if overlay_value == 'None':
        #return {'display': 'none'}, {'display': 'none'}
        return {'color': 'rgb(158, 158, 158)'}, {'color': 'rgb(158, 158, 158)'}
    else:
        return {}, {}

@app.callback(
    [Output('drop-target', 'value'), Output('drop-bin', 'value')],
    [Input('drop-overlay', 'value')],
    [State('drop-target', 'value'), State('drop-bin', 'value')]
)
def show_placeholder(overlay_value, curr_target, curr_bin):
    if overlay_value == 'None':
        #return {'display': 'none'}, {'display': 'none'}  # to disappear completely
        return 'placeholder', 'placeholder'
    else:
        new_target = 'prediction' if curr_target == 'placeholder' else curr_target
        new_bin = 'No bin' if curr_bin == 'placeholder' else curr_bin
        out = (new_target, new_bin)
        return out


# ----------------------------------------------------------------------------------------------------------------------
# Plot t-SNE embedding
@app.callback(
    Output('plot-tsne', 'figure'),
    [Input('submit-tsne', 'n_clicks')],
    [State('drop-layer', 'value'),
     State('drop-init', 'value'),
     State('drop-ndim', 'value'),
     State('input-lr', 'value'),
     State('input-perp', 'value'),
     State('input-niter', 'value')]
)
def update_plot_tsne(n_clicks, layer, init, ndim, lrate, perp, n_iter):
    globals()
    tsne_coord, labels, ids = tsne(model=net, dataloader=mydataloader, device=device, layer_feature=layer,
                                   ncomp=ndim, ini=init, perplex=perp, lr=lrate, niter=n_iter)
    # Create a different trace for each class
    traces = []
    for classe in classes:
        idx = np.where(labels==classe)
        if ndim==2:
            traces.append(go.Scattergl(x = tsne_coord[idx, 0].squeeze(),
                                     y = tsne_coord[idx, 1].squeeze(),
                                     mode='markers',
                                     name=classe,
                                     text=ids[idx]
                                   ))
        elif ndim==3:
            traces.append(go.Scatter3d(x = tsne_coord[idx, 0].squeeze(),
                                       y = tsne_coord[idx, 1].squeeze(),
                                       z = tsne_coord[idx, 2].squeeze(),
                                       mode='markers',
                                       marker=dict(size=3),
                                       name=classe,
                                       text=ids[idx]
                                    ))
    if ndim==2:
        return {
            'data': traces,
            'layout': go.Layout(
                xaxis={'title': 't-SNE 1'},
                yaxis={'title': 't-SNE 2'},
                hovermode='closest'
            )
        }
    if ndim==3:
        return {
            'data': traces,
            'layout': go.Layout(
                go.Scene(
                  xaxis={'title': 't-SNE 1'},
                  yaxis={'title': 't-SNE 2'},
                  #zaxis={'title': 't-SNE 3'}
                ),
                hovermode='closest'
            )
        }


# ----------------------------------------------------------------------------------------------------------------------
# Plot measurement of hovered trajectory
@app.callback(
    Output('plot-meas', 'figure'),
    [Input('plot-tsne', 'hoverData'),
     Input('slider-yrange', 'value'),
     Input('check-xshow', 'value'),
     Input('check-xrange', 'value'),
     Input('drop-overlay', 'value'),
     Input('drop-target', 'value'),
     Input('drop-bin', 'value')]
)
def update_plot_meas(hoverData, yrange, xshow, xcheck, overlay, target, bin):
    if hoverData is not None:
        hovered_cell = hoverData['points'][0]['text']
        dff = df[df['ID'] == hovered_cell]
        if overlay == 'None':
        # Create one trace per measurement
            plot = {
                'data': [go.Scattergl(
                    x=dff[dff['variable'] == meas]['Time'],
                    y=dff[dff['variable'] == meas]['value'],
                    mode='lines+markers',
                    name=meas,
                    marker={'opacity': 0.75}
                ) for meas in measurement],
                'layout': {
                    'title': hovered_cell,
                    'xaxis':{'title': 'Time'},
                    'yaxis':{'title': measurement,
                             'range': [yrange[0],yrange[1]]}
                }
            }

        else:
            # Create overlay of colors
            # Rebuild measurement array from melted dataframe, much faster than retrieving in dataloader in big dataset
            hovered_series = dff.pivot(index='variable', columns='Time', values='value')
            hovered_series = hovered_series.reindex(measurement, axis=0)
            hovered_series = hovered_series.reindex([str(i) for i in sorted(hovered_series.columns.astype('int'))], axis=1)
            hovered_series = np.array(hovered_series)
            # Subtract preprocessing before network (same as in class_dataset)
            if isinstance(subtract_numbers, (int, float)):
                subtract_array = np.array([subtract_numbers] * hovered_series.shape[0])
            elif isinstance(subtract_numbers, list):
                subtract_array = np.array(subtract_numbers)
            subtract_array = np.ones_like(hovered_series) * subtract_array[:, None]
            hovered_series -= subtract_array
            hovered_series = torch.tensor(hovered_series).to(device)

            if 'DF_CROP' in globals():
                start = int(DF_CROP.loc[DF_CROP['ID']==hovered_cell]['Crop_start'])
                end = int(DF_CROP.loc[DF_CROP['ID'] == hovered_cell]['Crop_end'])
                hovered_series = hovered_series[:, start:end]

            if overlay == 'CAM':
                ov_array = create_cam(model=net_unit_batch, device=device, array_series=hovered_series, target_class=target)
            elif overlay == 'Guided Backprop':
                ov_array = create_gbackprop(model=net_unit_batch, device=device, array_series=hovered_series, target_class=target)

            cmap = 'Jet'
            cmin, cmax = np.min(ov_array), np.max(ov_array)
            if bin == 'Li Threshold':
                thresh = threshold_li(ov_array)
                ov_array = np.where(ov_array >= thresh, 1, 0).astype('float')
                cmap = [[0,'grey'], [1,'red']]
                cmin, cmax = 0, 1
            elif bin == 'Mean Threshold':
                thresh = threshold_mean(np.abs(ov_array))
                ov_array = np.where(np.abs(ov_array) >= thresh, 1, 0).astype('float')
                cmap = [[0,'grey'], [1,'red']]
                cmin, cmax = 0, 1

            # Padding with NAs to compensate cropped time points and have overlay start and end in the right x positions, no pad in y
            if len(ov_array.shape) == 1:
                ov_array = np.reshape(ov_array, (1,-1)) # add dummy dimension for pad function for univariate series
            ov_array = np.pad(ov_array, [(0,0),
                                (int(DF_CROP.loc[DF_CROP['ID']==hovered_cell]['Crop_start']),
                                 int(len(dff)/len(measurement)) - int(DF_CROP.loc[DF_CROP['ID']==hovered_cell]['Crop_end']))],
                         'constant', constant_values=np.NaN)

            plot = {
                'data': [go.Scattergl(
                    x=dff[dff['variable'] == meas]['Time'],
                    y=dff[dff['variable'] == meas]['value'],
                    mode='lines+markers',
                    name=meas,
                    marker={'color': ov_array[imeas, :], 'colorscale': cmap, 'cmin':cmin, 'cmax':cmax,
                            'showscale': True, 'opacity': 0.75,}
                ) for imeas, meas in enumerate(measurement)],
                'layout': {
                    'title': hovered_cell,
                    'xaxis':{'title': 'Time'},
                    'yaxis':{'title': measurement,
                             'range': [yrange[0],yrange[1]]}
                }
            }

        # Add vertical lines to bound part of the signal seen by the network
        if 'DF_CROP' in globals():
            if xshow:
                x_start = int(DF_CROP.loc[DF_CROP['ID']==hovered_cell]['Crop_start'] + start_time)
                x_end =  int(DF_CROP.loc[DF_CROP['ID'] == hovered_cell]['Crop_end'] + start_time) - 1
                plot['layout']['shapes'] = [
                    {
                        'type': 'line',
                        'xref': 'x',
                        'yref': 'paper',
                        'x0': x_start,
                        'y0': 0,
                        'x1': x_start,
                        'y1': 1,
                        'opacity': 0.7,
                        'line': {'color': 'red', 'width': 2.5, 'dash': 'dash'}
                    },
                    {
                        'type': 'line',
                        'xref': 'x',
                        'yref': 'paper',
                        'x0': x_end,
                        'y0': 0,
                        'x1': x_end,
                        'y1': 1,
                        'opacity': 0.7,
                        'line': {'color': 'red', 'width': 2.5, 'dash': 'dash'}
                    }
                ]
            if xcheck:
                plot['layout']['xaxis'].update({'range': [x_start, x_end]})
    else:
        # Dummy placeholder plot for initialization, otherwise doesn't update upon subsequent hover
        plot = {
            'data': [go.Scattergl(
                x=[0],
                y=[min_slider],
                mode='lines+markers',
                name=meas,
                marker={'opacity': 0.75}
            ) for meas in measurement],
            'layout': {
                'title': '',
                'xaxis':{'title': 'Time'},
                'yaxis':{'title': measurement,
                         'range': [yrange[0],yrange[1]]}
            }
        }
    return plot



# ----------------------------------------------------------------------------------------------------------------------
# Display probability of classification of hovered trajectory
@app.callback(
    Output('table-proba', 'children'),
    [Input('plot-tsne', 'hoverData')]
)
def update_table_proba(selected_id):
    def generate_table(dataframe, hilight_class, max_rows=10):
        return html.Table(
            # Header
            [html.Tr([html.Th(col) if col != hilight_class else html.Th(col, style={'background-color': 'red'}) for col in dataframe.columns])] +

            # Body
            [html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))]
        )
    if selected_id is not None:
        cell_id = selected_id['points'][0]['text']
        cell_class = classes_dict[df.loc[df['ID'] == cell_id]['Class'].unique()[0]]
        filtered_df = round(DF_PROBS.loc[[cell_id]], 4)
        return generate_table(filtered_df, cell_class)


# ----------------------------------------------------------------------------------------------------------------------
# Export selected trajectories
@app.callback(
    Output('button-export', 'style'),
    [Input('button-export', 'n_clicks')],
    [State('plot-tsne', 'selectedData'), State('button-export', 'style')]
)
def export_selection(nclicks, selected_points, curr_style):
    # Dummy return
    button_style = curr_style
    if nclicks > 0:
        selected_points = selected_points['points']
        selected_ids = [point['text'] for point in selected_points]
        dff = deepcopy(df[df['ID'].isin(selected_ids)])  # copy avoids pandas warning about modifying copy
        dff['column_wide'] = dff['variable'] + '_' + dff['Time'].astype('object')
        col_order = [meas + '_' + str(ti) for meas in measurement for ti in sorted(dff['Time'].unique().astype('int'))]
        # Rebuild measurement array from melted dataframe, much faster than retrieving in dataloader in big dataset
        dff_wide = dff.pivot(index='ID', columns='column_wide', values='value')
        dff_wide = dff_wide.reindex(col_order, axis=1)

        # ----------------------------------------------
        # Add crop, tsne coords and model output
        frame_class = dff[['ID', 'Class']].groupby('ID').first().reset_index()
        frame_class.rename(columns={'Class': 'Class_ID'}, inplace=True)
        frame_class['Class'] = frame_class['Class_ID']
        frame_class.replace({'Class': classes_dict}, inplace=True)

        frame_coord = {point['text']: (round(point['x'], 4), round(point['y'], 4)) for point in selected_points}
        frame_coord = pd.DataFrame.from_dict(frame_coord, orient='index', columns=['xTSNE', 'yTSNE'])
        frame_coord.reset_index(inplace=True)
        frame_coord.rename(columns={'index': 'ID'}, inplace=True)

        frame_probs = deepcopy(DF_PROBS.loc[selected_ids])
        frame_probs = round(frame_probs, 4)
        frame_probs.columns = ['Prob_' + col for col in frame_probs.columns]
        frame_probs.reset_index(inplace=True)
        frame_probs.rename(columns={'index': 'ID'}, inplace=True)

        frame_feats = deepcopy(DF_FEATS.loc[selected_ids])
        frame_feats = round(frame_feats, 4)
        frame_feats.reset_index(inplace=True)
        frame_feats.rename(columns={'index': 'ID'}, inplace=True)

        frame_crop = deepcopy(DF_CROP[DF_CROP['ID'].isin(selected_ids)])
        # ----------------------------------------------

        df_out = [frame_class, frame_coord, frame_probs, frame_feats, frame_crop, dff_wide]
        df_out = reduce(lambda left,right: pd.merge(left, right, on='ID'), df_out)
        df_out.set_index('ID', inplace=True)

        # Open contextual menu to save
        root = Tk()
        root.filename = filedialog.asksaveasfilename(initialdir=".", title="Export selection",
                                                     filetypes=(("csv files","*.csv"), ("all files","*.*")),
                                                     initialfile='table_export.csv')
        path_to_save = root.filename
        root.destroy()

        # Dummy return
        # If press cancel in context menu, do nothing
        if path_to_save == ():
            return button_style
        else:
            df_out.to_csv(path_or_buf=path_to_save)
            return button_style
    else:
        return button_style


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=False, port=8051)
