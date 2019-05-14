from torch.nn import functional as F
import numpy as np

# todo: longest strides to largest areas/volumes


def create_cam(model, device, dataloader=None, id_series=None, array_series=None, feature_layer='features',
               clip=1.5, target_class='prediction'):
    """
    Create class activation map either by providing a series id and looping through loader
    until the id is found, or directly by providing the series as numerical array.
    If none is provided, but a dataloader is, will just pick next trajectory there

    :param model: variable containing the neural net.
    :param device: device of model
    :param dataloader: must have batch size 1. In a sample, the series must be returned as 'series',
    identifier as 'identifier'.
    :param id_series: If provided, loop through loader to look for this series.
    :param array_series: Manually provide a sample, must be a tensor.
    :param feature_layer: Name of the last convolution layer.
    :param plot: Boolean.
    :param clip: Clip max value to n standard deviation
    :param target_class: Create a CAM for which class? If 'prediction', creates a CAM for the predicted class. Otherwise
    give index of the class.
    :return: CAM as a numpy array.
    """
    # Def and checks
    def returnCAM(feature_conv, weight_softmax, class_idx):
        bz, nc, l = feature_conv.shape
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, l)))
        cam = cam - np.min(cam)
        return cam
    if model.batch_size != 1:
        print("Batch size of model must be 1")
        return

    # Hook the average pooling output (last feature layer)
    feature_blobs = []
    def hook_feature(module, input, output):
        feature_blobs.append(output.cpu().data.numpy())
    model._modules.get(feature_layer).register_forward_hook(hook_feature)
    # Get weights associated with each average pool
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

    # Get series and id for plot
    series, id_series = select_series(dataloader, id_series, array_series, device, return_id=True)

    # Create CAM
    logit = model(series)
    h_x = F.softmax(logit, dim=1).data.squeeze()
    if target_class == 'prediction':
        # Return CAM for predicted class
        probs, idx = h_x.sort(dim=0, descending=True)
        CAM = returnCAM(feature_blobs[0], weight_softmax, [idx[0].item()]).squeeze()
    else:
        CAM = returnCAM(feature_blobs[0], weight_softmax, target_class).squeeze()

    # Clip high values to improve map readability
    if clip:
        np.clip(CAM, a_min=None, a_max=np.mean(CAM)+ clip*np.std(CAM), out=CAM)

    return CAM


def BinarizedCam(model, array_series, percmax=0.5, feature_layer='features', device='cuda',
                 clip=0, target_class='prediction'):
    """
    Modified version of CAM to make it similar to smoothgrad
    """
    if percmax > 1 or percmax < 0:
        raise ValueError('"percmax" must be between 0 and 1')
    cam = create_cam(model, array_series=array_series, feature_layer=feature_layer,
                     device=device, clip=clip, target_class=target_class)
    threshold = percmax * np.max(cam)
    cam[np.where(cam <= threshold)] = 0
    cam[np.where(cam > threshold)] = 1
    return cam


def returnCAM(feature_conv, weight_softmax, class_idx):
    bz, nc, l = feature_conv.shape
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, l)))
    cam = cam - np.min(cam)
    return cam


def extend_bin(array, max_ext, direction='both'):
    """Extend regions of 1D binarized array."""
    # Spot regions with derivatives and extend n times
    ext_array = np.array(array).copy()
    for i in range(max_ext):
        diff = np.diff(ext_array)
        right_ind = (np.where(diff == -1)[0] + 1,)  # back to tuple to allow direct array indexing
        left_ind = np.where(diff == 1)
        if direction=='right' or direction=='both':
            ext_array[right_ind] = 1
        if direction=='left' or direction=='both':
            ext_array[left_ind] = 1
    return ext_array


def longest_strides(array, k=None, value=1):
    """Return the k longest strides with value in the array."""
    # Make sure it's a numpy array
    array = np.array(array).copy()
    indicator = np.where(array==value, 1, 0)
    # make sure all runs of ones are well-bounded
    bounded = np.hstack(([0], indicator, [0]))
    # get 1 at run starts and -1 at run ends
    diff = np.diff(bounded)
    run_starts, = np.where(diff > 0)
    run_ends, = np.where(diff < 0)
    lengths = run_ends - run_starts
    # reorder to have longest strides last
    order = np.argsort(lengths)
    lengths = lengths[order]
    run_starts = run_starts[order]
    run_ends = run_ends[order]
    if k:
        lengths = lengths[-k:]
        run_starts = run_starts[-k:]
        run_ends = run_ends[-k:]
    out = {(run_starts[i], run_ends[i]): lengths[i] for i in range(len(lengths))}
    return out


def select_series(dataloader=None, id_series=None, array_series=None, device=None, return_id=True):
    """
    Used in create_*_maps to select a series either from a dataloader with ID or directly use provided series. Can also
    provide a dataloader without ID to simply pick up next series in the loader.
    :return: The series properly formatted
    """
    flag_series = True
    if id_series is not None and array_series is not None:
        raise ValueError('At most one of "id_series" and "array_series" can be provided.')

    # If the series is provided as in ID, loop through loader until found
    if id_series:
        # go to list because typically provided as string but pytorch batch convert to list
        id_series = [id_series]
        if dataloader.batch_size != 1:
            print("Size of dataloader must be 1")
            return
        for sample in dataloader:
            if sample['identifier'] == id_series:
                series = sample['series']
                series = series.to(device)
                # uni: batch, 1 dummy channel, length TS
                # (1,1,length) for uni; (1,1,2,length) for bi
                series = series.view((1,) + series.shape)
                flag_series = False
                break
        # If not found
        if flag_series:
            print('ID not found in the dataloader')
            return

    if array_series is not None:
        series = array_series
        series = series.double()
        series = series.to(device)
        # uni: batch, 1 dummy channel, length TS
        # (1,1,length) for uni; (1,1,2,length) for bi
        series = series.view((1,1,) + series.shape)
        id_series = "Series manually provided"
        flag_series = False

    if flag_series:
        sample = next(iter(dataloader))
        series, correct_class, id_series = sample['series'], sample['label'], sample['identifier']
        print("When sampling from dataloader, take the actual class of the sample instead of input.")
        series = series.to(device)
        # uni: batch, 1 dummy channel, length TS
        # (1,1,length) for uni; (1,1,2,length) for bi
        series = series.view((1,) + series.shape)

    if return_id:
        return series, id_series
    else:
        return series
