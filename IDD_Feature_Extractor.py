#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.models as models
import numpy as np
import time
from tqdm import tqdm


# In[2]:


device = ('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[3]:


class IDDMaskedDataset(Dataset):
    
    
    def __init__(self,img_path_list, transform=None):
        
        self.resize_size = (512,512)
        self.len = len(img_path_list)
        self.img_path_list = img_path_list
           

    def __getitem__(self, index):
        try:
            
            img_path = self.img_path_list[index]
            self.label = img_path.split(".png")[0].split("__")[1]
            self.img = Image.open(img_path)
            self.img = self.img.resize(self.resize_size)
            self.img = torch.from_numpy(np.array(self.img).transpose(2,0,1))
            self.label = torch.tensor(int(self.label))
   
            return self.img, self.label
            
        except IOError as e:
            print(e)
            return None
        
    
    def __len__(self):
        return self.len
    


# In[4]:


train_file = open("./train_imgs_idd_masked_file_list.txt", 'r')
file_path_list = [line.rstrip() for line in train_file.readlines()]
print(f'There are {len(file_path_list)} images process')
dataset = IDDMaskedDataset(file_path_list)


# In[5]:


dataset_loader = DataLoader(dataset=dataset, batch_size=384, num_workers=4, shuffle=False) 


# In[6]:


resnet50 = models.resnet50(pretrained=True)
resnet50.to(device)
resnet50 = nn.DataParallel(resnet50)

resnet50.eval()

values_dict = {}
start_time = time.time()
for data, target in tqdm(dataset_loader):
    data = data.to(device,dtype=torch.float)
    target = target.to(device)
    
    with torch.no_grad():
        z = resnet50(data)
        z_list = z.tolist()
        target_list = target.tolist()
        
        values_dict.update(dict(zip(tuple(map(tuple,z.tolist())),target.tolist())))
        

print(f'Total processing time: {time.time() - start_time} seconds')
file_name = '/ssd_scratch/cvit/dksingh/idd_individual_features.pth'
torch.save(values_dict,file_name)


# #### Total time for cityscapes: 5090.432156801224 seconds
