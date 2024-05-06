import os
import parser
import sys
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import argparse
# from pandas_ml import ConfusionMatrix
import torch.nn.functional as F
# from model import resnet50
# from models.getModel import get_encoder
from resnet import resnet50
import random
import numpy as np
from sklearn.metrics import confusion_matrix
# from semodel import resnet50
import matplotlib.pyplot as plt
from torch.nn import DataParallel
import torch.nn.init as nn_init
from os import path
from PIL import Image
import numpy as np
import pandas as pd
import datetime
import time
from fusion_multi_model import Build_MultiModel_szzyy_pair_onlyimg
from dataloader.image_transforms import Image_Transforms
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 计算回归指标的函数
def compute_regression_metrics(predictions, targets):
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mse)
    corr = r2_score(targets, predictions)
    return mse, mae, rmse, corr

def writefile(name, list):
    # print(list)

    f = open(name+'.txt', mode='w')
    for i in range(len(list)):
        s = str(list[i]).replace('{', '').replace('}', '').replace("'", '').replace(':', ',') + '\n'
        f.write(s)
    f.close()



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
        # image_list=self.im_names[idx].split(';')
        # # img_list = [os.path.join(self.im_dir,str(self.im_path_head[idx]), string) for string in image_list]
        # img_list = [os.path.join(self.im_dir, string) for string in image_list]
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






def main():
    parser = argparse.ArgumentParser()
    gpuid=2
    parser.add_argument("--task_id", "-id", type=str, default="2", help="5fold id")
    parser.add_argument("--fusion_type", "-mt", type=str, default="avg", help="fusion type  avg linear")
    parser.add_argument("--backbone", "-bk", type=str, default="ResNet50", help="backbone ResNet50 ResNet50BAM  ResNet50CBAM  ResNet50SE  Alexnet")
    args = parser.parse_args()
    taskname=args.task_id
    savename=taskname
    print(taskname)
    backbone=args.backbone
    print(backbone)
    #avg linner

    fusion_type=args.fusion_type
    print(fusion_type)
    path='/data/new_with_line/'+backbone+'/'+fusion_type+'/'+savename

    mkdir(path)
    save_path_best = path+'/'+taskname+'_best.pth'
    device = torch.device("cuda:"+str(gpuid) if torch.cuda.is_available() else "cpu")

    print("using {} device.".format(device))
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    label_path='/data/Test5_resnert_szzyy/'
    IMAGE_PATH = '/data/gaowh/data/files/szzyy2/312new_with_up_down_line90p/'


    TRAIN_LISTS = ['Fold'+taskname+'_train.csv']
    TEST_LISTS = ['Fold'+taskname+'_test.csv']

    CLASSES = ['AGE','GXY','TNB','GXZ','TXBGAS','GNSXZ','SMOKE','DRINK']


    batch_size = 8
    print(batch_size)

    # 4 load dataset
    train_transforms = Image_Transforms(mode='train', dataloader_id=1, square_size=256, crop_size=224).get_transforms()
    val_transforms = Image_Transforms(mode='val', dataloader_id=1, square_size=256, crop_size=224).get_transforms()


    # Create training and test loaders
    validate_loader = load_data(label_path, TEST_LISTS, IMAGE_PATH,CLASSES, batch_size, val_transforms,'test')

    train_loader = load_data(label_path, TRAIN_LISTS, IMAGE_PATH,CLASSES, batch_size, train_transforms,'train')



    dfres = pd.read_csv(path_join(label_path, TEST_LISTS[0]))
    val_num=dfres.shape[0]

    dftrain = pd.read_csv(path_join(label_path, TRAIN_LISTS[0]))

    train_num=dftrain.shape[0]



    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    net = Build_MultiModel_szzyy_pair_onlyimg(backbone=backbone,fusion_type=fusion_type)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net, device_ids=[gpuid])

    net.to(device)
    loss_function = nn.CrossEntropyLoss()


    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    # optimizer = optim.Adam(params, lr=0.0001)

    optimizer = optim.Adam(params, lr=0.0001, amsgrad=True, weight_decay=0.0001)


    Loss_list = []
    Loss_list_val = []

    Accuracy_list = []
    Accuracy_list_val = []


    epochs =80
    best_f1 = 0.0
    train_steps = len(train_loader)


    val_steps=len(validate_loader)

    start_time=time.time()

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0

        train_acc = 0.0  #

        val_loss=0.0

        train_bar = tqdm(train_loader)


        for count, (data, target,_) in enumerate(train_bar):

            optimizer.zero_grad()
            # newimg=np.concatenate((images, images), axis=2) # axes are 0-indexed, i.e. 0, 1, 2

            # non_im_input = non_im_input.to(device)
            target = target.to(device)

            # im_extra=im_extra.to(device)
            target=target.view(-1)

            logits = net(data)

            # logits = net(images.to(device))  #这个就是结果
            loss = loss_function(logits, target.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

            predict_y = torch.max(logits, dim=1)[1]

            train_acc += torch.eq(predict_y, target.to(device)).sum().item()

        train_accurate = train_acc/train_num
        Accuracy_list.append(train_accurate)


        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch

        predict_all=[]
        gt_all=[]
        predictions = []
        labels = []


        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images,val_labels,_ = val_data

                val_labels=val_labels.view(-1)

                outputs = net(val_images)
                # probs = F.softmax(outputs, dim=1)

                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                loss_val = loss_function(outputs, val_labels.to(device))
                val_loss+=loss_val.item()

                gt_all.extend(val_labels.tolist())
                predict_all.extend(predict_y.tolist())

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num

        # f1 = f1_score(val_labels, outputs, average="weighted", labels=[0])

        report = classification_report(gt_all, predict_all)
        print(report)

        f1 = f1_score(gt_all, predict_all, average='macro')
        print("macro F1:"+str(f1))

        Loss_list_val.append(val_loss / val_steps)

        Accuracy_list_val.append( val_accurate)
        Loss_list.append(running_loss / train_steps)

        cm = confusion_matrix(np.array(gt_all),np.array(predict_all))
        print("Confusion Matrix:")
        print(cm)


        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        # if f1 > best_f1:
        #     best_f1 = f1
        #     torch.save(net.state_dict(), save_path_best)



    print('Finished Training')





if __name__ == '__main__':
    main()