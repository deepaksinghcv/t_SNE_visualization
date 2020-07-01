#!/usr/bin/env python
# coding: utf-8

# In[1]:


import multiprocessing
import numpy as np
from PIL import Image
import uuid
import time


# In[2]:


def delabeler(path_tuple):
    '''
    Takes a tuple which consists of original RGB image and the labelIds image
    Generate individually labeled RGB images
    NOTE: The generated file has a random uuid string and the class label joined wiht __ (double underscores)
    as the file name
    '''
    
    root_path = '/ssd_scratch/cvit/dksingh/idd/expanded_images/'
    orig_img_path = path_tuple[0]
    label_img_path = path_tuple[1]
    
    orig_img = Image.open(orig_img_path)
    label_img = Image.open(label_img_path)
    
    uniq_label_ids = set(np.unique(label_img))
    blacklisted_label_ids = {2,5,18,27,28,34,35,36,37,38,39}
    filtered_label_ids = uniq_label_ids - blacklisted_label_ids
    
    
    
    for label_id in filtered_label_ids:
        chosen_label_mask = np.array(label_img) == label_id
        chosen_label_mask_stack = np.stack((chosen_label_mask, chosen_label_mask, chosen_label_mask), axis=-1)
        chosen_label_mask_img = np.array(orig_img) * chosen_label_mask_stack
        
        random_file_name = uuid.uuid1().hex+"__"+str(label_id)+".png"
        
        temp_img = Image.fromarray(chosen_label_mask_img)
        temp_img.save(root_path+random_file_name)
        
#         print(f"saved: {random_file_name}")    


# #### To test a sample

# In[3]:


# path_tuple= ('/ssd_scratch/cvit/dksingh/cityscapes/leftImg8bit/train/cologne/cologne_000149_000019_leftImg8bit.png','/ssd_scratch/cvit/dksingh/cityscapes/gtFine/train/cologne/cologne_000149_000019_gtFine_labelIds.png')
# delabeler(path_tuple)


# ### Generate the path list

# In[4]:


def get_label_image_path(img_path):
    return img_path.replace("leftImg8bit","gtFine",1)[0:-15]+"gtFine_labelids.png"

file_pointer = open("./train_imgs_idd_file_list.txt", 'r')
img_path_list = [line.rstrip() for line in file_pointer.readlines()]
list_of_path_tuples = [(img_path, get_label_image_path(img_path)) for img_path in img_path_list]


# ### Start the multiprocessing

# In[5]:


start_time = time.time()
pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
pool.map(delabeler,list_of_path_tuples)
pool.close()
print(f'Total generation time: {time.time()-start_time} seconds')


# #### Cityscapes train took 6075 seconds.
# #### IDD train took ___ seconds

# In[ ]:




