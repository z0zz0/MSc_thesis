import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np

class AdniImagesDataset(Dataset):
    ### root - Should contain root folder of all the sub-directories and files. Example, data/AD/xxxx.nii or data/MCI/xxxx.nii.
    ###        AdniImagesDataset only accepts .nii files and the classes will be based on the sub-directories e.g., AD.
    ###        Note that the folder containing the data must be organized as follows, root/class/filename.extension
    
    def __init__(self, root, transforms=None, unique_subjects=False):
        self.transforms = transforms
        self.unique_subjects = unique_subjects
        
        self.images_list, self.targets = self.__getImgsAndTargets(root)
        
    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        data = self.images_list[idx]
        target = self.targets[idx]
        if self.transforms:
            self.transforms(data)
        
        return data, target


########################## HELPER FUNCTIONS ########################


    def __standardize(self, img):
        mean = np.mean(img)
        std = np.std(img)
        return (img-mean)/std
    
    
    def __getImgsAndTargets(self, root):
        imgs = []
        targets = []
        classes = self.__getClasses(root)
        counts = {}
        for c in classes:
            counts[c] = 0
            
        if self.unique_subjects:
            for image_path in self.__getPathToUniqueSubjectScans(root, classes):
                imgs.append( torch.from_numpy( self.__standardize( np.asarray(nib.load(image_path).dataobj).reshape((1,160,160,96)) ) ).float() ) 
                c = image_path.split("/")[2]
                targets.append( classes[c] )
                counts[c] += 1
        else:
            for image_path in self.__getPaths(root):
                imgs.append( torch.from_numpy( self.__standardize( np.asarray(nib.load(image_path).dataobj).reshape((1,160,160,96)) ) ).float() )
                c = image_path.split("/")[2]
                targets.append( classes[c] )
                counts[c] += 1
            
        print(f"Counts = {counts}")
        #return imgs, torch.nn.functional.one_hot( torch.tensor(targets) )
        return imgs, torch.tensor(targets) # No need to OneHot encode if using CrossEntropyloss
    
    
    def __getClasses(self, root):
        classes = {}
        for val, key in enumerate(next(os.walk(root))[1]):
            classes[key] = val
        print(f"Classes = {classes}")
        return classes
    
    
    def __getPaths(self, root):
        lst = []
        for root, dirs, files in os.walk(root):
            for file in files:
                if file.endswith(".nii") and len(dirs) == 0: # dirs is now part of 'root' string and dont include dirs in path since it'll show as an empty list []
                    lst.append(f'{root}/{file}')
        return lst
    
    
    def __getPathToUniqueSubjectScans(self, root, classes):
        imgs = {}
        path = self.__getPaths(root)
        paths = {}
        for i, image_path in enumerate(path):
            tmp = image_path.split("_")
            key = f"{tmp[1]}_{tmp[2]}_{tmp[3]}_{tmp[4]}"
            
            if i == 758: ## specifically at this index of all the paths, the name varies slightly, thus index 14 is needed for date
                new_date = int(tmp[14])
            else:
                new_date = int(tmp[16])

            if key in imgs:    
                old_date = imgs[key]
                if old_date > new_date:
                    imgs[key] = new_date
                    paths[key] = image_path
            else:
                imgs[key] = new_date
                paths[key] = image_path

        ## If there is unequal amount of cases (which there is), remove so there always are equal amount of cases
        balanced_cases_paths = []
        paths = list(paths.values())
        counts = {}

        for case in classes:
            counts[case] = 0

        for path in paths:
            case = path.split("/")[2]
            counts[case] += 1

        min_num_case = min(counts.values())
        for d in counts:
            counts[d] -= min_num_case

        for path in paths:
            case = path.split("/")[2]
            if counts[case] > 0:
                counts[case] -= 1
            else:
                balanced_cases_paths.append(path)

        return balanced_cases_paths


########################## HELPER FUNCTIONS END ########################