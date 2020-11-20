import sys
import pkg_resources

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse as sp

import umap.umap_ as umap
import hdbscan
from scipy import linalg

import torch
import torchvision
import torch.utils.data as data_utils


from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scipy.stats import spearmanr

from .utils import *

PACKAGE_PATH = pkg_resources.resource_filename('DGCyTOF', "/")

def validate_model(model_fc, val_tensor, classes, params_val = {'batch_size':10000, 'shuffle': False, 'num_workers': 6}):
    '''
    
        Runs validation on the validation dataset, print out the performance of the trained model for all cell types and returns 
        them as a zip.

        **Params**:

        * model_fc: Trained PyTorch model, must have a forward function and utilize argmax as classification in its design
        * val_tensor: Validation dataset as a Torch tensor.
        * classes: List of types of cells
        * params_val: dictionary containing information for dataloader, requires at least a batch_size, shuffle, and num_workers
        keys.
            * batch_size: Number of data points in a single batch, default 128
            * shuffle: Shuffle the batches prior to training, default True
            * num_workers: Number of processes that will be used to load data, default 6

        **Returns**:

        * Zip of listed results. Each respective row contains pred,label,out in validation_results
            * pred: Predicted label of a data point
            * label: Actual label of a data point
            * out: Output value of the data running forward through model_fc

    '''
    assert (len(set(classes)) > 1), "There must be at least 2 classes"
    
    labels = len(classes)
        
    model_fc.eval()
    
    val_loader = data_utils.DataLoader(dataset = val_tensor, **params_val)
    
    class_correct = list(0. for i in range(labels))
    class_total = list(0. for i in range(labels))

    val_correct = 0
    val_total = 0

    for data in val_loader:
        val_samples, val_labels = data
        val_outputs = model_fc(Variable(val_samples))
        _, val_predicted = torch.max(val_outputs.data, 1)   # Find the class index with the maximum value.
        c = (val_predicted == val_labels).squeeze()
        for i in range(val_labels.shape[0]):
            label = val_labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

        val_total += val_labels.size(0)
        val_correct += (val_predicted == val_labels).sum()

    print("Accuracy:", round(100 *val_correct.item() / val_total, 4))
    print('-'*100)
    for i in range(labels):
        print('Accuracy of {} : {}'.format (
            classes[i], round(100 * class_correct[i] / class_total[i], 3)))
        
    # Return. validation results
    return list(zip(val_predicted, val_labels, val_outputs))

        
def train_model(model_fc, X_train, max_epochs = 20, params_train = {'batch_size':128, 'shuffle': True, 'num_workers': 6}):
    '''
    
        Trains the entered deep learning model using Pytorch. Utulized criterion is CrossEntropyLoss and optimizer is Adam 
        optimizer with learning rate 0.001. The input model is set to evaluation mode after training.

        **Params**:

        * model_fc: PyTorch model, must have a forward function and utilize argmax as classification in its design
        * max_epochs: Number of epochs the model will be trained for 
            * default: 20
        * params_train: dictionary containing information for trainloader, requires at least a batch_size, shuffle, and 
        num_workers keys.
            * batch_size: Number of data points in a single batch, default 128
            * shuffle: Shuffle the batches prior to training, default True
            * num_workers: Number of processes that will be used to load data, default 6

        **Returns**:

        * None: Nothing is returned, the model is set to eval mode after training. 

    '''
    train_loader = data_utils.DataLoader(dataset = X_train, **params_train)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_fc.parameters(), lr=0.001)

    for epoch in range(max_epochs):  # loop over the dataset multiple times

        total_loss = 0
        total_correct = 0

        for data in train_loader: # Get Batch
            samples, labels = data 

            preds = model_fc(samples) # Pass Batch
            loss = criterion(preds, labels) # Calculate Loss

            optimizer.zero_grad() # Zero Gradients       
            loss.backward() # Calculate Gradients
            optimizer.step() # Update Weights

            #loss_outputs.append(outputs)
            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)

        print("epoch", epoch, "total_correct:", total_correct, "loss:", loss.item(), "total_loss:", total_loss)
    model_fc.eval()
    
def preprocessing(dataset, columns_to_remove = []):
    '''
    
        Removes a set of columns if specified, returns input CYTOF data with labels, the labels as a list, and the unlabeled input 
        CyTOF data.

        **Params**:

        * dataset: CyTOF Data Matrix
        * columns_to_remove: List of columns to remove from the dataset
            * default: Empty list

        **Returns**:

        * X_data_labeled: input CYTOF data with labels
        * y_data: the labels as a list
        * data_unlabeled: non-classified CyTOF data
    
    '''
    data = dataset.drop(columns_to_remove, axis = 1) # removing unnecessary columns
    data_labeled = data[data.label.notnull()]     # Labeled data
    X_data_labeled = data_labeled.drop(['label'], axis = 1) # labeled data without labels
    y_data = data_labeled['label']    # labels of labeled data
    data_unlabeled = data[data.label.isnull()].drop(['label'], axis = 1)
    
    return X_data_labeled, y_data, data_unlabeled
    
        
        
def calibrate_data(model_fc, X_test, classes, validation_results, unlabeled_data):
    '''
    
        Calibrates the test set of the data based on the training model and validation results in performing over the test set. 
        Test data either are classified more accurately or labeled as a new subtype based on the minimum threshold 
        in classification. Minumum threshold is computed by obtaining the lowest correlation probability from validation results.
        Returns calibrated incorrect data.

        **Params**:

        * model_fc: Trained PyTorch model, must have a forward function and utilize argmax as classification in its design
        * X_test: CyTOF data for test set.
        * classes: List of types of cells
        * validation_results: zip created by validate_model. Contains the following keys:
            * pred: Predicted label of a data point
            * label: Actual label of a data point
            * out: Output value of the data running forward through model_fc

        **Returns**:

        * updated_incorrect_data: Calibrated incorrect data.
    
    '''
    
    correct_pred_info = [(label, pred, np.max(F.softmax(out, dim=0).data.numpy())) for pred,label,out in validation_results if (pred == label)]
    corr_prob = [prob for label, pred, prob in correct_pred_info]
    
    test_outputs = model_fc(Variable(X_test))
    _, test_predicted = torch.max(test_outputs.data, 1)   # Find the class index with the maximum value. 

    output_logits = F.softmax(test_outputs, dim=1).data.numpy()
    output_labels = test_predicted.data.numpy()
    probabilities = np.max(output_logits, axis=1)
    tem = [round(i,4) for i in probabilities]
    #incorrect_index = [tem.index(a) for a in tem if a <= min(corr_prob)]
    correct_index = [i for i,v in enumerate(tem) if v > min(corr_prob)]
    incorrect_index = [i for i,v in enumerate(tem) if v <= min(corr_prob)]
    incorrect_data = pd.DataFrame([X_test[i].data.numpy() for i in incorrect_index])
    print("A total of " + str(len(incorrect_data)) + "  incorrect labeling as been found")
    
    ##################
    
    test_with_labels = unlabeled_data.copy()
    test_with_labels['label'] = test_predicted.numpy()
    test_with_labels = test_with_labels.reset_index(drop=True)
    test_correct_with_labels = test_with_labels.iloc[correct_index]
    
    test_correct_list = dict()

    for subtype_no in range(0, len(classes)):
        test_correct_list[subtype_no] = test_correct_with_labels[test_correct_with_labels.label == subtype_no].drop(['label'], 
                                                                                                  axis = 1).values.tolist()
    
    incorrect_list = incorrect_data.values.tolist()
    
    rho_dict = dict()
    
    for i in range(len(classes)):
        if i == 1:
            rho_dict[i], _ = spearmanr(np.transpose(test_correct_list[i][-12000:] + incorrect_list))
        else:
            rho_dict[i], _ = spearmanr(np.transpose(test_correct_list[i] + incorrect_list))

    #################
    
    wrongly_incorrect_index = dict()
    wrongly_incorrect_corr = dict()

    for subtype_no in range(0, len(classes)):
        if subtype_no == 1:
            wrongly_incorrect_index[subtype_no], wrongly_incorrect_corr[subtype_no], _ = active_learning_index(
            test_correct_list[subtype_no][-12000:], rho_dict[subtype_no])
        else:
            wrongly_incorrect_index[subtype_no], wrongly_incorrect_corr[subtype_no], _ = active_learning_index(
            test_correct_list[subtype_no], rho_dict[subtype_no])
            
    ###################
    
    temp = pd.DataFrame(np.zeros((len(incorrect_list), len(classes))))
    for subtype_no in range(0, len(classes)):
        temp[subtype_no][wrongly_incorrect_index[subtype_no]] = wrongly_incorrect_corr[subtype_no]
    temp_1 = temp.loc[(temp != 0).any(1)] # Drop rows with all zeros 
    
    # Setting highest value in row to 1 and rest to 0

    m = np.zeros_like(temp_1.values)
    m[np.arange(len(temp_1)), temp_1.values.argmax(1)] = 1

    temp_2 = pd.DataFrame(m, columns = temp_1.columns, index = temp_1.index ).astype(int)
    
    updated_test = dict()

    for subtype_no in range(0, len(classes)):
        # indices (of cells) that should be added to respective celltype
        temp_index = np.array(temp_2[subtype_no].index[temp_2[subtype_no].to_numpy().nonzero()])
        updated_test[subtype_no] = test_correct_list[subtype_no] + incorrect_data.iloc[temp_index].values.tolist()
        
    updated_incorrect_data = incorrect_data.drop(np.array(temp_2.index), axis=0)
    
    return (updated_incorrect_data)

def dimensionality_reduction_and_clustering(input_data, n_neighbors = 5, min_dist = 0.01):
    
    """
    
        Reduce the dimensions of the input data using HDBBSCAN + UMAP. While running, also displays the 2D clustered data 
        (clusterPlot).
        
        **Params**:

        * input_data: CyTOF matrix data
        * n_neighbors: Number of neighbors set to cluster data points
            * Default: 5
        * min_dist: Minimum distance between embeded points, utilized in the UMAP function.
            * Default: 0.01

        **Returns**:

        * data_umap: Data with UMAP transform to 2 dimensions.
        * y_HDBSCAN_umap: HDBSCAN + UMAP of the input data
        * new_subtypes: Returns found new subtypes
        * umapPlot: 2D plot of data with new subtypes

    """

    data_umap = u_map(input_data, 2, n_neighbors, min_dist)
    
    y_HDBSCAN_umap = hdbscan.HDBSCAN(min_samples=1, min_cluster_size=10).fit_predict(data_umap)
    
    unique, counts = np.unique(y_HDBSCAN_umap, return_counts=True)
    new_cells_dict = dict(zip(unique, counts))
    
    new_celltypes = []
    for i in new_cells_dict.keys(): 
        new_celltypes.append("New Subtype {}".format(i+2))
    
    clusterPlot = Dim_Red_Plot('UMAP and HDBSCAN',data_umap, y_HDBSCAN_umap+1, list(new_cells_dict.keys()), new_celltypes)
    
    return data_umap, y_HDBSCAN_umap, new_celltypes, clusterPlot
        
def Dim_Red_Plot(name, data, labels, no_classes, class_names):
    
    '''
    
        Creates a 2D plot of the data, displays it and returns it.
        
        **Params**:

        * name: Name of the plot 
        * data: 2 dimensional embedding of data, for instance UMAP transformed data
        * no_classes: List of classes
        * class_names: List of class names

        **Returns**:

        * figure: Plt plot
        
    '''
    
    fig = plt.figure(figsize=(7,4))
    plt.scatter(np.array(data)[:, 0], np.array(data)[:, 1], s=10, cmap='gist_rainbow', c=labels, alpha=0.5)
    
    cbar = plt.colorbar(boundaries=np.arange(len(no_classes)+1)-0.5)
    cbar.set_ticks(np.arange(len(no_classes)))
    cbar.set_ticklabels(class_names)
    cbar.ax.tick_params(labelsize=14)
    
    plt.xlabel('Reduced Axis 1', fontsize=15)
    plt.ylabel('Reduced Axis 2', fontsize=15)
    plt.title('{c}'.format(c = name), fontsize=18)
    
    return fig


def Dim_Red_Plot_3d (data, labels, all_celltypes):
    '''
    
        Creates a 3D plot of the data, displays it and returns it.
        
        **Params**:

        * data: Low dimension embedding of data, for instance UMAP transformed data
        * labels: List of labels corresponding to each cell
        * all_celltypes: List of celltype names, where each celltype is in its respective label position

        **Returns**:

        * figure: Plt plot
        
    '''
    fig = plt.figure(figsize=(15,12))
    ax = Axes3D(fig)
    X = np.arange(-25, 22, 2)
    Y = np.arange(-22, 22, 2)
    X, Y = np.meshgrid(X, Y)
    
    ax.scatter(np.array(data)[:, 0], np.array(data)[:, 1], labels, c=labels, alpha=0.5, cmap=plt.cm.get_cmap('gist_ncar', 29))

    cmap = mpl.cm.gist_ncar
    
    ax.set_zticks(range(1, len(set(all_celltypes)) + 1))
    a = ax.get_zticks().tolist()
    a = all_celltypes
    ax.set_zticklabels(a)

    ax.set_xlabel('Reduced Axis 1', fontsize=14)
    ax.set_ylabel('Reduced Axis 2', fontsize=14)
    # ax.set_zlabel('t-SNE 3')
    # ax.set_zlabel('Label', fontsize=14)
    plt.show()
    
    return fig