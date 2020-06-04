import numpy as np
from scipy.ndimage import label
import torch
from torch.nn import functional as F
from torch.nn import ReLU


def create_gbackprop(model, device, target_class, dataloader=None, id_series=None, array_series=None, clip = 0):
    """
    Create class activation map either by providing a series id and looping through loader
    until the id is found, or directly by providing the series as numerical array.
    If none is provided, but a dataloader is, will just pick next trajectory there

    :param model: variable containing the neural net.
    :param device: device of model.
    :param target_class: Backprop for which class? If 'prediction'.
    :param dataloader: must have batch size 1. In a sample, the series must be returned as 'series',
    identifier as 'identifier'.
    :param id_series: If provided, loop through loader to look for this series.
    :param array_series: Manually provide a sample, must be a tensor.
    :param clip: Clip max value to n standard deviation.
    :return: Saliency map as numpy array
    """
    # Pick series either from input, dataloader and ID, or next from data loader
    # ----------------------------------------------------------------------------
    series, id_series = select_series(dataloader, id_series, array_series, device, return_id=True)
    # ----------------------------------------------------------------------------
    # Modify the backpropagation through ReLU layers (guided backprop)
    def relu_hook_function(module, grad_in, grad_out):
        """If there is a negative gradient, changes it to zero"""
        if isinstance(module, ReLU):
            return (torch.clamp(grad_in[0], min=0.0),)

    # Loop through layers, hook up ReLUs with relu_hook_function
    # backward hook will modify gradient in ReLU during backprop
    hook_idx = 0
    for pos, module in model.features._modules.items():
        if isinstance(module, ReLU):
            # Use unique names for each hook in order to be able to remove them later
            hook_name = "hook" + str(hook_idx)
            exec(hook_name + "= module.register_backward_hook(relu_hook_function)")
            hook_idx += 1

    # Create saliency map
    # Start recording operations on input
    series.requires_grad_()
    model.batch_size = 1
    output = model(series)
    model.zero_grad()
    # Target for backprop
    one_hot_output = torch.FloatTensor(1, output.shape[-1]).zero_()
    one_hot_output[0][target_class] = 1
    one_hot_output = one_hot_output.double()
    one_hot_output = one_hot_output.to(device)
    # Vanilla Backprop
    output.backward(gradient=one_hot_output)
    # Gradients wrt inputs are the saliency map
    saliency = series.grad.squeeze().cpu().numpy()
    if clip:
        saliency = np.clip(saliency, a_min=None, a_max = np.mean(saliency)+clip*np.std(saliency))

    # Remove hooks from model
    for idx in range(hook_idx):
        hook_name = "hook" + str(idx)
        exec(hook_name + ".remove()")
    return saliency


def create_cam(model, device, dataloader=None, id_series=None, array_series=None, feature_layer='features',
               clip=0, target_class='prediction'):
    """
    Create class activation map either by providing a series id and looping through loader
    until the id is found, or directly by providing the series as numerical array.
    If none is provided, but a dataloader is, will just pick next trajectory there

    :param model: variable containing the neural net.
    :param device: device of model.
    :param dataloader: must have batch size 1. In a sample, the series must be returned as 'series',
    identifier as 'identifier'.
    :param id_series: If provided, loop through loader to look for this series.
    :param array_series: Manually provide a sample, must be a tensor.
    :param feature_layer: Name of the last convolution layer.
    :param clip: Clip max value to n standard deviation.
    :param target_class: Create a CAM for which class? If 'prediction', creates a CAM for the predicted class. Otherwise
    give index of the class.
    :return: CAM as a numpy array.
    """
    # Def and checks
    def returnCAM(feature_conv, weight_softmax, class_idx):
        """
        Perform CAM computation: use weights of softmax to weight individual filter response in the filter layer.
        feature_conv: output of last convolution before global average pooling.
        weight_soft_max: array with all softmax weights
        class_idc: index of the class for which to produce the CAM.
        """
        # Batch size, number channels (features, number of filters in convolution layer),
        # height (nber measurements), width (length measurements)
        bz, nc, h, w = feature_conv.shape
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam - np.min(cam)
        cam = cam.reshape(h,w)
        return cam
    def returnCAM1D(feature_conv, weight_softmax, class_idx):
        """
        Special case of CAM when input has only one measurement. Identical to returnCAM except for shape that has one
        less dimension.
        """
        # Batch size, number channels, length
        bz, nc, l = feature_conv.shape
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, l)))
        cam = cam - np.min(cam)
        return cam
    if model.batch_size != 1:
        raise ValueError('Batch size of model must be 1')

    # Hook the layer output before average pooling (last feature layer)
    feature_blobs = []
    def hook_feature(module, input, output):
        feature_blobs.append(output.cpu().data.numpy())
    model._modules.get(feature_layer).register_forward_hook(hook_feature)
    # Get weights associated with each average pool element
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
        if len(feature_blobs[0].shape) == 3:
            CAM = returnCAM1D(feature_blobs[0], weight_softmax, [idx[0].item()]).squeeze()
        elif len(feature_blobs[0].shape) > 3:
            CAM = returnCAM(feature_blobs[0], weight_softmax, [idx[0].item()]).squeeze()
    else:
        if len(feature_blobs[0].shape) == 3:
            CAM = returnCAM1D(feature_blobs[0], weight_softmax, target_class).squeeze()
        elif len(feature_blobs[0].shape) > 3:
            CAM = returnCAM(feature_blobs[0], weight_softmax, target_class).squeeze()

    # Clip high values to improve map readability
    if clip:
        np.clip(CAM, a_min=None, a_max=np.mean(CAM) + clip*np.std(CAM), out=CAM)

    return CAM


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
    """Return the k longest segments of 1s in a binary array. Structure must be a valid argument of
    scipy.ndimage.label. By default, segments can be connected vertically and horizontally, pass appropriate structure
    for different behaviour. Output is a dictionary where values are the size of the segment and keys are tuples that
    indicate all the positions of a segment, just like numpy.where(). So can use the keys to directly subset an numpy
    array at the positions of the segments."""
    assert np.all(np.isin(array, [0,1]))
    # Label each segment with a different integer, 0s are NOT labeled (i.e. remain 0)
    array_segments, num_segments = label(array, structure=structure)
    label_segments, size_segments = np.unique(array_segments, return_counts=True)
    # Special case when only 1s in the array
    if not np.all(array==1):
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


def extract_pattern(origin_array, coord_tuple, NA_fill = True):
    """
    Extract a pattern from an array via its list of coordinates stored in a tuple (as returned by np.where() or
    longest_segments()). The pattern has rectangular shape, with NA padding if NA_fill is True. This is useful to export
    patterns in 2 or more dimensions and plot them/compute distances between them.
    :param coord_tuple: a tuple of coordinates as returned by np.where(). For example ((x1,x2,x3), (y1,y2,y3)).
    :param origin_array: an array from which to extract the pattern.
    :param NA_fill bool, whether to fill parts of the rectangle not listed in coord_tuple. IF False, will use values
    from origin_array.
    :return: a rectangular 2D numpy array with the pattern, padded with NAs. Number of rows from origin_array is
    maintained.
    """
    assert len(origin_array.shape) == 1 or len(origin_array.shape) == 2
    assert len(origin_array.shape) == len(coord_tuple)
    if NA_fill:
        out = np.full_like(origin_array, np.nan)
        if len(origin_array.shape) == 1:
            out[coord_tuple] = origin_array[coord_tuple]
            out = out[np.min(coord_tuple[1]) : (np.max(coord_tuple[1]) + 1)]
        elif len(origin_array.shape) == 2:
            out[coord_tuple] = origin_array[coord_tuple]
            out = out[:, np.min(coord_tuple[1]) : (np.max(coord_tuple[1]) + 1)]

    elif len(origin_array.shape) == 1:
        out = origin_array[np.min(coord_tuple) : (np.max(coord_tuple)+1)]
    elif len(origin_array.shape) == 2:
        out = origin_array[:, np.min(coord_tuple[1]) : (np.max(coord_tuple[1])+1)]

    return  out


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
