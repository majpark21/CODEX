from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from functools import reduce
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import warnings


def model_output(model, dataloader, export_prob=True, export_feat=True, softmax=True, batch_size = None, device=None,
                 feature_layer='pool'):
    """
    Function to get representation and probability of each trajectory in a loader.
    model: str or nn
    """

    # Default arguments and checks
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if batch_size is None:
        batch_size = dataloader.batch_size
    if dataloader.batch_size != batch_size:
        raise(ValueError('batch_size must match the one of the DataLoader,'
                         ' reset DataLoader with batch size: {},'
                         ' or leave None to use DataLoader batch size.'.format(batch_size)))

    if dataloader.drop_last:
        warnings.warn('drop_last==TRUE, in DataLoader, some data might be discarded.')
    else:
        for i in dataloader:
            size_last_batch = len(i['identifier'])
        if size_last_batch % batch_size != 0:
            raise(ValueError('Batch size: {}, is not a multiple of the number of elements in the DataLoader: {}.'
                             ' Make it multiple or set drop_last=True in the DataLoader,'
                             ' but with the latter last batch will be discarded.'.format(batch_size,
                                                                    (len(dataloader)-1)*batch_size + size_last_batch)))
    # path to model file
    if isinstance(model, str):
        model = torch.load(model)
        model.eval()
        model.double()
        model.batch_size = batch_size
        model = model.to(device)
    else:
        model.eval()
        model.double()
        model.batch_size = batch_size
        model = model.to(device)

    # Lists with results of every batch
    lidt = []
    llab = []
    if export_prob:
        lprob = []
    if export_feat:
        lfeat = []
        def hook_feature(module, input, output):
            lfeat.append(output.data.squeeze().cpu().numpy())
        model._modules.get(feature_layer).register_forward_hook(hook_feature)

    for sample in iter(dataloader):
        image_tensor, label, identifier = sample['series'], sample['label'], sample['identifier']
        image_tensor = image_tensor.to(device)
        # uni: batch, 1 dummy channel, length TS
        # (1,1,length) for uni; (1,1,2,length) for bi
        assert len(dataloader.dataset[0]['series'].shape) == 2
        nchannel, univar_length = dataloader.dataset[0]['series'].shape
        if nchannel == 1:
            view_size = (model.batch_size, 1, univar_length)
        elif nchannel >= 2:
            view_size = (model.batch_size, 1, nchannel, univar_length)
        image_tensor = image_tensor.view(view_size)

        scores = model(image_tensor)
        if softmax:
            scores = F.softmax(scores, dim=1).data.squeeze()
        # Store batch results
        llab.append([label.data.squeeze().cpu().numpy()])
        lidt.append(identifier)
        if export_prob:
            lprob.append(scores.data.squeeze().cpu().numpy())

    frames = []
    # 1) Frame with ID
    lidt = np.concatenate(lidt)
    llab = np.concatenate(llab)
    frames.append(pd.DataFrame(list(zip(lidt, llab)), columns=['ID', 'Class']))
    # 2) Frame with proba
    if export_prob:
        nclass = lprob[0].shape[1] if batch_size > 1 else len(lprob[0])
        lprob = np.vstack(lprob)
        colnames = ['Prob_' + str(i) for i in range(nclass)]
        frames.append(pd.DataFrame(lprob, columns=colnames))
    # 3) Frame with features
    if export_feat:
        nfeats = lfeat[0].shape[1] if batch_size > 1 else len(lfeat[0])
        lfeat = np.vstack(lfeat)
        colnames = ['Feat_' + str(i) for i in range(nfeats)]
        frames.append(pd.DataFrame(lfeat, columns=colnames))
    df_out = pd.concat(frames, axis=1)
    return df_out


#%%
def tensorboard_to_df(path_to_logs, tags):
    """
    Read tensorboard logs and return a DataFrame with values. Useful for further plotting.
    :param path_to_logs: path to directory with log files
    :param tag: string name of the tag to read and export. Can pass a string or a list of strings. Tags must have
    common step vector in the logs.
    :return: A pandas Dataframe in wide format with one column for each tag
    """

    if isinstance(tags, str):
        tags = [tags]
    dfList = []
    event_acc = EventAccumulator(path_to_logs)
    event_acc.Reload()
    for tag in tags:
        w_times, step_nums, vals = zip(*event_acc.Scalars(tag))
        df = pd.DataFrame(data={'step': step_nums, tag: vals})
        dfList.append(df)

    # Merge all dataframes
    out = reduce(lambda x, y: pd.merge(x, y, on='step'), dfList)
    return out