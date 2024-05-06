
from sklearn.metrics import roc_curve, auc


# import pandas as pd
# from os import *
import torch.nn.functional as F
import os
import sys
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


from resnet import resnet50

import numpy as np


import matplotlib.pyplot as plt

from torch.nn import DataParallel

import torch.nn.init as nn_init

from os import path
from PIL import Image
import numpy as np
import pandas as pd


from fusion_multi_model import Build_MultiModel_szzyy_pair_onlyimg

from dataloader.image_transforms import Image_Transforms

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import numpy as np
from sklearn.metrics import confusion_matrix
from pycm import *
import argparse
import random
from pandas_ml import ConfusionMatrix



def mkdir(path):
    import os

    path = path.strip()
    path = path.rstrip("\\")

    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        print(path + ' success')
        return True
    else:
        return False



path_join = path.join

def get_random_images(image_files, max_img, seed=222):
    total_frames = len(image_files)
    if total_frames <= max_img:
        if seed is not None:
            random.seed(seed)
        output = image_files.copy()
        if seed is not None:
            random.shuffle(output)
        while len(output) < max_img:
            output += image_files
        return sorted(output[:max_img])
    
    if seed is not None:
        random.seed(seed)

    indices = random.sample(range(total_frames), max_img)
    indices.sort()
    image_list = [image_files[i] for i in indices]
    return image_list

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, im_dir, im_names, im_labels,im_path,im_transforms=None):
        self.im_dir = im_dir
        self.im_labels = im_labels
        self.im_names = im_names
        self.im_path_head=im_path

        if im_transforms:
            self.im_transforms = im_transforms
        else:
            self.im_transforms = transforms.Compose([transforms.ToTensor()])
    def __len__(self):
        return len(self.im_labels)
    def __getitem__(self, idx):
        img_list=get_random_images( self.im_names[idx].split(';'),2)
        img_list = [os.path.join(self.im_dir, string) for string in img_list]
        images = []
        for image_path in img_list:
            try:
                im = Image.open(image_path).convert('RGB')
                im = self.im_transforms(im)
                images.append(im)
            except:
                print('Error: Failed to open or verify image file {}'.format(image_path))
        return images, self.im_labels[idx], self.im_path_head[idx]


 

def load_data(label_path, train_lists, img_path,classes,
              batchsize, im_transforms,type):
    train_sets = []
    train_loaders = []
    for train_list in train_lists:
        full_path_list = path_join(label_path,train_list)
        df = pd.read_csv(full_path_list)
        im_names = df['images'].to_numpy()

        im_labels=df['grade'].to_numpy()
        im_path=df['id'].to_numpy()

        train_sets.append(CustomDataset(img_path, im_names, im_labels , im_path,im_transforms))
        train_loaders.append(torch.utils.data.DataLoader(train_sets[-1], batch_size=batchsize, shuffle=True,num_workers=8))
        print('Size for {0} = {1}'.format(train_list, len(im_names)))
    return train_loaders[0]


parser = argparse.ArgumentParser()
parser.add_argument("--task_id", "-id", type=str, default="2", help="5fold id")
parser.add_argument("--fusion_type", "-mt", type=str, default="avg", help="fusion type")    
parser.add_argument("--backbone", "-bk", type=str, default="ResNet50", help="backbone   option ResNet50   ")    

args = parser.parse_args()


backbone=args.backbone
print(backbone)

fusion_type=args.fusion_type
print(fusion_type)

taskname=args.task_id
print(taskname)

thispath='/data/test_res/ResNet50/avg/2/'
weightpath='/data/test_res/ResNet50/avg/2/2_best.pth'
label_path='/data/'
IMAGE_PATH = '/data/data/files/szzyy/312new/'
TRAIN_LISTS = ['Fold'+taskname+'_train.csv']
TEST_LISTS = ['Fold'+taskname+'_test.csv']
CLASSES = ['AGE','GXY','TNB','GXZ','TXBGAS','GNSXZ','SMOKE','DRINK']

train_transforms = Image_Transforms(mode='train', dataloader_id=1, square_size=256, crop_size=224).get_transforms()
val_transforms = Image_Transforms(mode='val', dataloader_id=1, square_size=256, crop_size=224).get_transforms()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 32

# Create training and test loaders
validate_loader = load_data(label_path, TEST_LISTS, IMAGE_PATH,CLASSES, batch_size, val_transforms,'test')

train_loader = load_data(label_path, TRAIN_LISTS, IMAGE_PATH,CLASSES, batch_size, train_transforms,'train')




model = Build_MultiModel_szzyy_pair_onlyimg(backbone=backbone,fusion_type=fusion_type)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=[0])

 
model.to(device)

model.load_state_dict(
    torch.load(weightpath))

model.eval()
predictions = []
labels = []
headid=[]

# 对测试集进行推断
with torch.no_grad():
    for val_images,val_labels,im_path_head in validate_loader:
        outputs = model(val_images)
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)
        
        predictions.extend(probs[:, 1].tolist())
        labels.extend(val_labels.tolist())
        headid.extend(im_path_head.tolist())

# 计算TPR和FPR
fpr, tpr, thresholds = roc_curve(labels, predictions)
data = pd.DataFrame({'Labels': labels, 'Predictions': predictions})

data.to_csv(thispath+taskname+'.csv', index=False)

threshold = 0.5
predictions = np.array(predictions)


binary_predictions = np.where(predictions > threshold, 1, 0)

conf_matrix = ConfusionMatrix(actual_vector=labels, predict_vector=binary_predictions)

filename = thispath+ "confusion_matrix.txt"

delimiter = "\t"

matrix_str = str(conf_matrix)

with open(filename, "w") as f:
    f.write(matrix_str)


auc_score = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
# plt.show()
plt.savefig(thispath+'roc_curve.png')

data = {'headid': headid, 'prediction': predictions, 'label': labels}
df = pd.DataFrame(data)

df.to_csv(thispath+'predictions_labels.csv', index=False)
