##################################################################################################
# Inspect output of the classifier: confusion table, accuracy... Plot tops classification sample #
##################################################################################################

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader
from class_dataset import myDataset, ToTensor, Subtract, RandomCrop
from torch.nn import functional as F
from itertools import repeat
import scipy.cluster.hierarchy
from load_data import DataProcesser
from torchvision import transforms
import pandas as pd
import os

# Todo: get rid of for loop for acc_per_class (batch size = size dataset)

def confusion_table(model, dataloader, classes, device):
    """
    Return a confusion table of classificatuin
    """
    confusion={k:{k:0 for k in classes} for k in classes}
    for sample in iter(dataloader):
        image_tensor, label = sample['series'], classes[sample['label'].item()]
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
        logit = model(image_tensor)
        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(dim=0, descending=True)
        prediction = classes[idx[0]]
        confusion[label][prediction] += 1
    return confusion


def acc_per_class(model, dataloader, classes, device):
    """
    Return accuracy of classification / class. Batch size (1)
    """
    dic_correct_count = {k:0 for k in classes}
    dic_total_count = {k:0 for k in classes}

    for sample in iter(dataloader):
        image_tensor, label = sample['series'], classes[sample['label'].item()]
        image_tensor = image_tensor.to(device)

        # uni: batch, 1 dummy channel, length TS
        # (1,1,length) for uni; (1,1,2,length) for bi
        image_tensor = image_tensor.view((1,) + image_tensor.shape)

        logit = model(image_tensor)
        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(dim=0, descending=True)

        prediction = classes[idx[0]]
        dic_total_count[label] += 1
        if prediction == label:
            dic_correct_count[label] += 1

    dic_accuracy = {k: round(dic_correct_count[k]/dic_total_count[k], 3) for k in classes}
    out = {'accuracy':dic_accuracy, 'total_count': dic_total_count, 'correct_count': dic_correct_count}
    return out


def top_classification_perclass(model, dataloader, classes, device, n=10):
    """
    Returns the top n correct classification / perclass (largest confidence)
    """
    best_class = {k: list(repeat((0, 'init_label'), n)) for k in classes}
    for sample in iter(dataloader):
        image_tensor, label = sample['series'], classes[sample['label'].item()]
        image_tensor = image_tensor.to(device)
        # uni: batch, 1 dummy channel, length TS
        # (1,1,length) for uni; (1,1,2,length) for bi
        image_tensor = image_tensor.view((1,) + image_tensor.shape)
        logit = model(image_tensor)
        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(dim=0, descending=True)
        prediction = classes[idx[0]]

        if prediction == label:
            # sorted list such that 0 is the smallest in each class, proba is 0-element of the tuple
            low_prob = best_class[label][0][0]
            if probs[0] >= low_prob:
                del best_class[label][0]
                best_class[label].append((probs[0].item(), sample['identifier']))
                best_class[label].sort(key=lambda tup: tup[0])

    return best_class


def worst_classification_perclass(model, dataloader, classes, device, n=10):
    """
    Returns the top n incorrect classification / perclass (largest confidence)
    Check errors that were made with high confidence, could be informative
    to see which patterns were wrongly recognized
    """
    worst_class = {k: list(repeat((0, 'init_label', 'pred_label'), n)) for k in classes}
    for sample in iter(dataloader):
        image_tensor, label = sample['series'], classes[sample['label'].item()]
        image_tensor = image_tensor.to(device)
        # uni: batch, 1 dummy channel, length TS
        # (1,1,length) for uni; (1,1,2,length) for bi
        image_tensor = image_tensor.view((1,) + image_tensor.shape)
        logit = model(image_tensor)
        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(dim=0, descending=True)
        prediction = classes[idx[0]]

        if prediction != label:
            # sorted list such that 0 is the smallest in each class, proba is 0-element of the tuple
            low_prob = worst_class[label][0][0]
            if probs[0] >= low_prob:
                del worst_class[label][0]
                worst_class[label].append((probs[0].item(), sample['identifier'], prediction))
                worst_class[label].sort(key=lambda tup: tup[0])

    return worst_class


def top_scoring_perclass(model, dataloader, classes, device, n=10):
    """
    Returns trajectories with highest score for each class, independently of the classification being correct or not.
    """
    # Scores at not lower bounded so set minimum very low...
    top_class = {k: list(repeat((-1e6, 'init_label', 'pred_label'), n)) for k in classes}
    for sample in iter(dataloader):
        image_tensor, label = sample['series'], classes[sample['label'].item()]
        image_tensor = image_tensor.to(device)
        # uni: batch, 1 dummy channel, length TS
        # (1,1,length) for uni; (1,1,2,length) for bi
        image_tensor = image_tensor.view((1,) + image_tensor.shape)
        scores = model(image_tensor)
        h_x = F.softmax(scores, dim=1).data.squeeze()
        probs, idx = h_x.sort(dim=0, descending=True)
        prediction = classes[idx[0]]

        # Convert score to dict
        scores = scores.data.squeeze().cpu().numpy()
        scores = list(zip(classes, scores))

        for classe,score in scores:
            # current lowest score in the class (ordered list)
            low_score = top_class[classe][0][0]
            if score >= low_score:
                del top_class[classe][0]
                top_class[classe].append((score.item(), sample['identifier'], prediction))
                top_class[classe].sort(key=lambda tup: tup[0])

    return top_class


def visualize_layer(model, layer_idx=0, linkage='average'):
    weights = model.features[layer_idx].cpu().weight.detach().numpy().squeeze()
    nfilt, h ,w = weights.shape
    link = scipy.cluster.hierarchy.linkage(weights.reshape(nfilt, -1), method=linkage)
    order = scipy.cluster.hierarchy.dendrogram(link)['leaves']
    for i in range(nfilt):
        plt.subplot(nfilt, 1, i+1)
        plt.imshow(weights[order[i]])
    plt.tight_layout()
    plt.show()
    return None


if __name__ == '__main__':
    data_file = 'data/synthetic_len750.zip'
    model_file = 'models/FRST_SCND/2019-05-31-19:30:05_synthetic_len750.pytorch'
    meas_var = ['FRST', 'SCND']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_top_worst = 5

    model = torch.load(model_file)
    model.eval()
    model.double()
    model.batch_size = 1
    model = model.to(device)

    data = DataProcesser(data_file)
    data.subset(sel_groups=meas_var, start_time=0, end_time=750)
    data.get_stats()
    #data.process(method='center_train', independent_groups=True)
    data.split_sets()
    classes = tuple(data.classes.iloc[:,1])

    data_test = myDataset(dataset=data.validation_set, transform=transforms.Compose([
        #RandomCrop(output_size=model.length, ignore_na_tails=True),
        #Subtract(data.stats['mu']['KTR']['train']),
        ToTensor()]))
    test_loader = DataLoader(dataset=data_test,
                             batch_size=1,
                             shuffle=True,
                             num_workers=4)


    accuracy = acc_per_class(model, test_loader, classes, device)
    conft = pd.DataFrame.from_dict(confusion_table(model, test_loader, classes, device))
    conft = conft.reindex(classes, axis=0)
    conft = conft.reindex(classes, axis=1)
    conft['Accuracy'] = pd.Series(accuracy['accuracy'])
    print(conft)

    tops = top_classification_perclass(model, test_loader, classes, device, n=n_top_worst)
    worsts = worst_classification_perclass(model, test_loader, classes, device, n=n_top_worst)

    #%%
    # Plot top trajectories in a pdf
    lplot=[]
    for classe in classes:
        fig = plt.figure(figsize=(20, 10), dpi=160)
        for id in tops[classe]:
            id = id[1][0]
            subset = data.validation_set.loc[data.validation_set['ID'] == id].iloc[0, 2:]
            subset = np.array(subset).astype('float')
            plt.plot(subset, label=id)
            plt.title(classe)
            plt.legend()
        #plt.show()
        lplot.append(fig)
        #plt.close()

    pp = PdfPages('output/' + '_'.join(meas_var) + '/tops_' + os.path.basename(model_file).rstrip('.pytorch') + '.pdf')
    for plot in lplot:
        pp.savefig(plot)
    pp.close()

    #%%
    # Plot worst trajectories in a pdf file
    lplot=[]
    for classe in classes:
        fig = plt.figure(figsize=(20, 10), dpi=160)
        for item in worsts[classe]:
            if item[1] == 'init_label':
                continue
            id = item[1][0]
            mistake = item[2]
            subset = data.validation_set.loc[data.validation_set['ID'] == id].iloc[0, 2:]
            subset = np.array(subset).astype('float')
            plt.plot(subset, label=id + ' - ' + mistake)
            plt.title(classe)
            plt.legend()
        #plt.show()
        lplot.append(fig)
        #plt.close()

    pp = PdfPages('output/' + '_'.join(meas_var) + '/worsts_' + os.path.basename(model_file).rstrip('.pytorch') + '.pdf')
    for plot in lplot:
        pp.savefig(plot)
    pp.close()