from torch.nn import functional as F
import numpy as np
from scipy.ndimage import label

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


def extend_segments1D(array, max_ext, direction='both'):
    """Extend regions of 1D binarized array."""
    # Spot regions with derivatives and extend n times
    assert len(array.shape) == 1
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


def extend_segments(array, max_ext, direction ='both'):
    """Extend regions of 1 or 2D binarized array. If 2D, each row will be extended independently."""
    assert len(array.shape) == 1 or len(array.shape) == 2
    if len(array.shape) == 1:
        ext_array = extend_segments1D(array, max_ext, direction)
    elif len(array.shape) == 2:
        ext_array = np.array(array).copy()
        for irow in range(array.shape[0]):
            ext_array[irow, :] = extend_segments1D(array[irow, :], max_ext, direction)
    return ext_array


def longest_segments(array, k=None, structure=None):
    """Return the k longest segments with 1s in a binary array. Structure must be a valid argument of
    scipy.ndimage.label. By default, segments can be connected vertically and horizontally, pass appropriate structure
    for different behaviour. Output is a dictionary where values are the size of the segment and keys are tuples that
    indicate all the positions of a segment, just like numpy.where(). So can use the keys to directly subset an numpy
    array at the positions of the segments."""
    assert np.all(np.unique(array) == np.array([0,1]))
    # Label each segment with a different integer, 0s are NOT labeled (i.e. remain 0)
    array_segments, num_segments = label(array, structure=structure)
    label_segments, size_segments = np.unique(array_segments, return_counts=True)
    # np.unique returns ordered values, so 0 is always first
    label_segments = np.delete(label_segments, 0)
    size_segments = np.delete(size_segments, 0)
    # Longest segments first, along with label
    sorted_segments = sorted(zip(size_segments, label_segments), reverse=True)
    if k:
        sorted_segments = sorted_segments[:k]
    # Need to convert np.where output to tuple for hashable
    out = {tuple(tuple(i) for i in np.where(array_segments == lab)): size for size, lab in sorted_segments}
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
                if len(series.shape) == 1:
                    series = series.view(1, len(series))
                # uni: batch, 1 dummy channel, length TS
                # (1,1,length) for uni; (1,1,2,length) for bi
                assert len(series.shape) == 2
                nchannel, univar_length = series.shape
                if nchannel == 1:
                    view_size = (1, 1, univar_length)
                elif nchannel >= 2:
                    view_size = (1, 1, nchannel, univar_length)
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
        if len(series.shape) == 1:
            series = series.view(1, len(series))
        # uni: batch, 1 dummy channel, length TS
        # (1,1,length) for uni; (1,1,2,length) for bi
        assert len(series.shape) == 2
        nchannel, univar_length = series.shape
        if nchannel == 1:
            view_size = (1, 1, univar_length)
        elif nchannel >= 2:
            view_size = (1, 1, nchannel, univar_length)
        series = series.view(view_size)
        id_series = "Series manually provided"
        flag_series = False

    if flag_series:
        sample = next(iter(dataloader))
        series, correct_class, id_series = sample['series'], sample['label'], sample['identifier']
        print("When sampling from dataloader, take the actual class of the sample instead of input.")
        series = series.to(device)
        # uni: batch, 1 dummy channel, length TS
        # (1,1,length) for uni; (1,1,2,length) for bi
        assert len(series.shape) == 2
        nchannel, univar_length = series.shape
        if nchannel == 1:
            view_size = (1, 1, univar_length)
        elif nchannel >= 2:
            view_size = (1, 1, nchannel, univar_length)
        series = series.view(view_size)

    if return_id:
        return series, id_series
    else:
        return series
