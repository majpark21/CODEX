##################################################################################################
# Inspect output of the classifier: confusion table, accuracy... Plot tops classification sample #
##################################################################################################

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader
from class_dataset import myDataset, ToTensor, Subtract, RandomCrop
from load_data import DataProcesser
from torchvision import transforms
import pandas as pd
import os
import re
from utils import model_output


def confusion_matrix(model, dataloader, device=None, labels_classes=None):
    """
    Return confusion matrix for all element in dataloader.

    :param model: str or pytorch model. If str, path to the model file.
    :param dataloader: pytorch Dataloader, classification output will be created for each element in the loader. Pay
    attention to the attribute drop_last, if True last batch would not be processed. Increase to lower computation time.
    :param device: str, pytorch device. If None will try to use cuda, if not available will use cpu.
    :param labels_classes: a dictionary or pandas.Series to rename classes indexes.
    :return: A pandas DataFrame, actual classes in rows, predicted classes in columns.
    """
    df_out = model_output(model, dataloader, export_prob=True, export_feat=False, device=device)
    prob_cols = [col for col in df_out.columns if col.startswith('Prob_')]
    df_out['Prediction_colname'] = df_out[prob_cols].idxmax(axis=1)  # returns name of columns
    df_out['Prediction'] = df_out['Prediction_colname'].str.replace('^Prob_', '').astype('int')
    confmat = pd.crosstab(df_out['Class'], df_out['Prediction'])
    if labels_classes is not None:
        confmat.rename(labels_classes, axis='index', inplace=True)
        confmat.rename(labels_classes, axis='columns', inplace=True)
    return confmat


def acc_per_class(model, dataloader, device=None, labels_classes=None):
    """
    Return accuracy of classification per class.

    :param model: str or pytorch model. If str, path to the model file.
    :param dataloader: pytorch Dataloader, classification output will be created for each element in the loader. Pay
    attention to the attribute drop_last, if True last batch would not be processed. Increase to lower computation time.
    :param device: str, pytorch device. If None will try to use cuda, if not available will use cpu.
    :param labels_classes: a dictionary or pandas.Series to rename classes indexes.
    :return: A pandas Series with accuracies per class, index contains the class names.
    """
    confmat = confusion_matrix(model, dataloader, device)
    tot_count = confmat.sum(axis=1)
    diag_count = confmat.values[[np.arange(confmat.shape[0])] * 2]
    acc = diag_count / tot_count
    if labels_classes is not None:
        confmat.rename(labels_classes, axis='index', inplace=True)
    return acc


def top_confidence_perclass(model, dataloader, n=10, mode ='highest', device=None, softmax=True, labels_classes=None):
    """
    Returns the results of classification with highest or lowest confidence per class.

    :param model: str or pytorch model. If str, path to the model file.
    :param dataloader: pytorch Dataloader, classification output will be created for each element in the loader. Pay
    attention to the attribute drop_last, if True last batch would not be processed. Increase to lower computation time.
    :param n: int, the number of trajectories to return per class.
    :param device: str, pytorch device. If None will try to use cuda, if not available will use cpu.
    :param softmax: bool, whether to apply softmax to before selecting th trajectories.
    :param mode: str, one of ['highest', 'lowest'].
    :param labels_classes: a dictionary or pandas.Series to rename classes indexes.
    :return: A pandas DataFrame with columns: 'ID', 'Class', 'Prob_XXX' where XXX is the class index as returned by
    the model.
    """
    assert mode in ['highest', 'lowest']
    out = []
    df_out = model_output(model, dataloader, export_prob=True, export_feat=False, softmax=softmax, device=device)
    for iclass in range(len((df_out['Class'].unique()))):
        sort_by = 'Prob_' + str(iclass)
        if mode == 'highest':
            out.append(df_out.loc[df_out['Class']==iclass].sort_values(by=sort_by).tail(n))
        elif mode == 'lowest':
            out.append(df_out.loc[df_out['Class']==iclass].sort_values(by=sort_by).head(n))
    out = pd.concat(out, axis=0)
    if labels_classes is not None:
        out['Class'].replace(labels_classes, inplace=True)
        old_labels = {col: re.search('\d+$', col) for col in out.columns.values}
        new_labels = {col: re.sub('\d+$', labels_classes[int(old_labels[col].group())], col)
                      for col in old_labels if old_labels[col]}
        out.rename(new_labels, axis='columns', inplace=True)
    return out


def worst_classification_perclass(model, dataloader, n=10, device=None, softmax=True, labels_classes=None):
    """
    Returns the worst classification per class. Worst classifications are defined as incorrect classification (i.e. the
    model predicted a class that is not the one of individual) with largest confidence.

    :param model: str or pytorch model. If str, path to the model file.
    :param dataloader: pytorch Dataloader, classification output will be created for each element in the loader. Pay
    attention to the attribute drop_last, if True last batch would not be processed. Increase to lower computation time.
    :param n: int, the maximum number of trajectories to return per class.
    :param device: str, pytorch device. If None will try to use cuda, if not available will use cpu.
    :param softmax: bool, whether to apply softmax to before selecting th trajectories.
    :param labels_classes: a dictionary or pandas.Series to rename classes indexes.
    :return: A pandas DataFrame with columns: 'ID', 'Class', 'Prob_XXX' where XXX is the class index as returned by
    the model.
    """
    out = []
    df_out = model_output(model, dataloader, export_prob=True, export_feat=False, softmax=softmax, device=device)
    prob_cols = [col for col in df_out.columns if col.startswith('Prob_')]
    df_out['Prediction_colname'] = df_out[prob_cols].idxmax(axis=1)  # returns name of columns
    df_out['Prediction'] = df_out['Prediction_colname'].str.replace('^Prob_', '').astype('int')
    df_out = df_out.reindex(columns=['ID', 'Class', 'Prediction', 'Prediction_colname'] + prob_cols)
    for classe in df_out['Class'].unique():
        # Cases where real class is different from the predicted one but where confidence is high for the predicted
        to_append = df_out.loc[(df_out['Class'] != df_out['Prediction']) &
                               (df_out['Class'] == classe)].copy()
        # Skip if no wrong classification for this class
        if to_append.shape[0] == 0:
            continue
        # Report value of predicted class on each row
        to_append['Prediction_confidence'] = to_append.lookup(to_append.index, to_append.Prediction_colname)
        to_append.sort_values(by='Prediction_confidence', inplace=True)
        to_append = to_append.tail(n)
        out.append(to_append)
    out = pd.concat(out, axis=0).drop(columns=['Prediction_colname', 'Prediction_confidence'])
    if labels_classes is not None:
        out['Class'].replace(labels_classes, inplace=True)
        out['Prediction'].replace(labels_classes, inplace=True)
        old_labels = {col: re.search('\d+$', col) for col in out.columns.values}
        new_labels = {col: re.sub('\d+$', labels_classes[int(old_labels[col].group())], col)
                      for col in old_labels if old_labels[col]}
        out.rename(new_labels, axis='columns', inplace=True)
    return out


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