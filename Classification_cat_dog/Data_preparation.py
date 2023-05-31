#%% Prepare cat and dog data by separating them into training and testing sets
import numpy as np
import os
import random as rnd

#%%


ori_cat_dir = '/Users/elvist/Downloads/kagglecatsanddogs_5340/PetImages/Cat'
ori_dog_dir = '/Users/elvist/Downloads/kagglecatsanddogs_5340/PetImages/Dog'

# Training folder

train_cat_dir = '/Users/elvist/PycharmProjects/CV Projects/Classification_cat_dog/train/cat'
test_cat_dir = '/Users/elvist/PycharmProjects/CV Projects/Classification_cat_dog/test/cat'


train_dog_dir = '/Users/elvist/PycharmProjects/CV Projects/Classification_cat_dog/train/dog'
test_dog_dir = '/Users/elvist/PycharmProjects/CV Projects/Classification_cat_dog/test/dog'
#%%
# Create folder
def create_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)

#%%
create_folder(train_cat_dir)
create_folder(test_cat_dir)
create_folder(train_dog_dir)
create_folder(test_dog_dir)

#%% 
_, _, files_cat = next(os.walk(ori_cat_dir))
_, _, files_dog = next(os.walk(ori_dog_dir))

rnd.shuffle(files_cat) # randomly shuffle the data
rnd.shuffle(files_dog)

number_cat = len(files_cat) # number of files
number_dog = len(files_dog)

num_train = 0.8 * number_dog
# %%
import shutil
def copy_files(files, ori_dir, train_dir, test_dir, num_tra):
    pos = 0
    for f in files:
        img = os.path.join(ori_dir, f)
        if pos < num_tra:
            dest_folder = train_dir
        else:
            dest_folder = test_dir
        destination_img = os.path.join(dest_folder, f)
        shutil.copy(img, destination_img)
        pos +=1

#%%
copy_files(files_cat, ori_cat_dir, train_cat_dir, test_cat_dir, num_train)
copy_files(files_dog, ori_dog_dir, train_dog_dir, test_dog_dir, num_train)
# %%
