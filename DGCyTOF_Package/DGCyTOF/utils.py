# File for non-user functions
import numpy as np
import umap.umap_ as umap

# Util function
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

# Util function
def active_learning_index(test_correct_list, rho, indices = True):
    wrongly_incorrect_index =[]   # indices of wrongly predicted incorrect
    correct_size = len(test_correct_list) # length of correct class
    rho_correct = rho[:correct_size, :correct_size] # correlation matrix of correct class
    rho_correct_array = rho_correct[np.triu_indices(correct_size, k = 1)]  # array of correlations of correct class
    rho_avg = np.mean(rho_correct_array)
    
    if indices == True:
        wrongly_incorrect_index = [(i-correct_size) for i in range(correct_size+1, len(rho)) 
                                   if np.mean(rho[i, :correct_size]) > rho_avg]
        wrongly_incorrect_corr = [np.mean(rho[i, :correct_size]) for i in range(correct_size+1, len(rho)) 
                                  if np.mean(rho[i, :correct_size]) > rho_avg]
        return wrongly_incorrect_index, wrongly_incorrect_corr, rho_avg
    else:
        return rho_avg
    
# Util function
def u_map(data, n_components, n_neighbors, min_dist):
    mapped = umap.UMAP(n_components = n_components, n_neighbors=n_neighbors, min_dist=min_dist)
    data_umap = mapped.fit_transform(data)
    return data_umap
    
