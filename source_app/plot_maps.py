import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import ReLU


# ----------------------------------------------------------------------------------------------------------------------
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
        class_idx: index of the class for which to produce the CAM.
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
    if target_class == 'prediction' or target_class == 'placeholder':
        # Return CAM for predicted class
        probs, idx = h_x.sort(dim=0, descending=True)
        target_class = int(idx[0].item())
        if len(feature_blobs[0].shape) == 3:
            CAM = returnCAM1D(feature_blobs[0], weight_softmax, target_class).squeeze()
        elif len(feature_blobs[0].shape) > 3:
            CAM = returnCAM(feature_blobs[0], weight_softmax, target_class).squeeze()
    else:
        if len(feature_blobs[0].shape) == 3:
            CAM = returnCAM1D(feature_blobs[0], weight_softmax, target_class).squeeze()
        elif len(feature_blobs[0].shape) > 3:
            CAM = returnCAM(feature_blobs[0], weight_softmax, target_class).squeeze()

    # Clip high values to improve map readability
    if clip:
        np.clip(CAM, a_min=None, a_max=np.mean(CAM)+ clip*np.std(CAM), out=CAM)

    return CAM


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
    if target_class == 'prediction':
        h_x = F.softmax(output, dim=1).data.squeeze()
        probs, idx = h_x.sort(dim=0, descending=True)
        target_class = idx[0].item()
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

# ----------------------------------------------------------------------------------------------------------------------
def create_saliency_vanilla(model, device, target_class, dataloader=None, id_series=None, array_series=None,
                            clip = 3):
    # Pick series either from input, dataloader and ID, or next from data loader
    # ----------------------------------------------------------------------------
    series, id_series = select_series(dataloader, id_series, array_series, device, return_id=True)
    # ----------------------------------------------------------------------------
    # Create saliency map
    # Start recording operations on input
    series.requires_grad_()
    model.batch_size = 1
    output = model(series)
    model.zero_grad()
    # Target for backprop
    one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
    one_hot_output[0][target_class] = 1
    one_hot_output = one_hot_output.double()
    one_hot_output = one_hot_output.to(device)
    # Vanilla Backprop
    output.backward(gradient=one_hot_output)
    # Gradients wrt inputs are the saliency map
    saliency = series.grad.squeeze().cpu().numpy()
    if clip:
        saliency = np.clip(saliency, a_min=None, a_max = np.mean(saliency)+clip*np.std(saliency))

    return saliency

# ----------------------------------------------------------------------------------------------------------------------
def create_saliency_guided(model, device, target_class, dataloader=None, id_series=None, array_series=None,
                            clip = 3):
    from torch.nn import ReLU
    # Pick series either from input, dataloader and ID, or next from data loader
    # ----------------------------------------------------------------------------
    series, id_series = select_series(dataloader, id_series, array_series, device, return_id=True)
    # ----------------------------------------------------------------------------
    # Modify the backpropagation through ReLU layers (guided backprop)
    def relu_hook_function(module, grad_in, grad_out):
        """
        If there is a negative gradient, changes it to zero
        """
        if isinstance(module, ReLU):
            return (torch.clamp(grad_in[0], min=0.0),)

    # Loop through layers, hook up ReLUs with relu_hook_function
    # backward hook will modify gradient in RELU during backprop
    hook_idx = 0
    for pos, module in model.features._modules.items():
        if isinstance(module, ReLU):
            # Use unique names for each hook in order to be able to remove them later
            hook_name = "hook"+str(hook_idx)
            exec(hook_name + "= module.register_backward_hook(relu_hook_function)")
            hook_idx += 1

    # Create saliency map
    # Start recording operations on input
    series.requires_grad_()
    model.batch_size = 1
    output = model(series)
    model.zero_grad()
    # Target for backprop
    one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
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


# ----------------------------------------------------------------------------------------------------------------------
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
