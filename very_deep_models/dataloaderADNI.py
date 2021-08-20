import random
import torch
import numpy as np
from numpy.random import default_rng

## These functions will help me get train, validation and test loaders. Depending on use case, function exists for fetching kfold-cv data aswell.
## This is implemented in a way to be used with a balanced dataset.

################### FUNCTIONS START ###################

def split_train_test(dataset, n_classes, test_ratio=0.2):
    if test_ratio >= 1.0 or test_ratio < 0:
        raise Exception(f"test_ratio={test_ratio} isn't compatible.")
        
    train_idx = []
    test_idx = []
    classes_idx = []
    class_size = int(len(dataset)/n_classes)
    class_test_size = int(test_ratio*class_size)
    
    ## Create list of lists of class indices.
    ## Each sub-list has class indices
    for i in range(n_classes):
        start_idx = i*class_size
        end_idx = start_idx+class_size
        classes_idx.append(list( range(start_idx, end_idx) ))
    
    ## Splits train and test indices 
    for cx in classes_idx:
        tmp_test = cx[:class_test_size]
        train_idx.extend( list(set(cx)-set(tmp_test)) )
        test_idx.extend(tmp_test)
        
    ## Return subsets
    return torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, test_idx)


## Returns training and validation loaders
## assumes balanced dataset
def get_kfCV_loaders(dataset, n_classes, k, batch_size, num_workers=4):
    if k<=1 and isinstance(k, int):
        raise Exception(f"CV with K-fold={k}, doesn't work.")
    if int(len(dataset)/k) == 0 and isinstance(k, int):
        raise Exception(f"CV with K-fold={k} and dataset size of {len(dataset)} are not compatible")
    if (k<=0 or k>=1) and not isinstance(k, int):
        raise Exception("Check your k value. Something wrong with it.")
    
    generator = default_rng()
    class_size = int(len(dataset)/n_classes)
    
    if isinstance(k, int):
        val_class_size = int(class_size/k)
    elif isinstance(k, float):
        val_class_size = int(class_size*k)
    else:
        raise Exception("Check your k value. Something wrong with it.")
        
    set_idx = np.arange(len(dataset), dtype=int)
    val_idx = np.array([], dtype=int)
    
    ## Create list of lists of class indices.
    ## Each sub-list has class indices
    for i in range(n_classes):
        start_idx = int(i*class_size)
        stop = start_idx + val_class_size
        val_idx = np.append(val_idx, generator.choice(np.arange(start_idx, stop, dtype=int), size=val_class_size, replace=False))
        
    train_idx = np.setxor1d(set_idx, val_idx)
    
    train_data = torch.utils.data.Subset(dataset, train_idx)
    val_data = torch.utils.data.Subset(dataset, val_idx)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle = False, num_workers = num_workers)
    
    #print(f"Train dataset size={len(train_loader.dataset)}. Validation dataset size={len(val_loader.dataset)}")
    
    return train_loader, val_loader

#################### FUNCTIONS END ####################