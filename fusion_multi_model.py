import torch
import torch.nn as nn
# from fusion.segating import SEGating
# from fusion.average import project,SegmentConsensus
# from fusion.segating import SEGating
# from fusion.segating import SEGating
# from fusion.nextvlad import NextVLAD
# from models.cv_models.swin_transformer import swin
# from models.cv_models.swin_transformer_v2 import swinv2
# # from models.cv_models.resnest import resnest50, resnest101
# from models.cv_models.convnext import convnext_tiny, convnext_small, convnext_base
# from resnet import resnet50,resnet152
from ConvNeXt import convnext_tiny as create_model
from resnet_att import resnet50bam, resnet50cbam,resnet50se,resnet50,resnet152,resnet34
from alexmodel import AlexNet
# from models.FIT_Net import FITNet
# from model3d import resnet
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
        
 
# img+img share backbone
class Build_MultiModel_szzyy(nn.Module):
    def __init__(self, backbone='resnest50', input_dim=2048, num_classes=2, pretrained_modelpath='None'):
        super().__init__()



        self.num_classes = num_classes


        self.backbone=backbone


        self.model = resnet50()
        self.model.fc = Identity()


        self.lg2 = torch.nn.Linear(in_features=(2048+8), out_features=self.num_classes)


        # self.vector=self.lg2.weight



        print('init model:', backbone)



    def forward(self, img,extradata):

        all=[]

        ydoutput = self.model(img)
        
        all.append(ydoutput)
        all.append(extradata)


        all_output = torch.cat(all, -1) # b, c1+c2
                # 在预测过程中保存向量值
   
        vector = all_output

        self.vector = vector.detach()

        res = self.lg2(all_output)


        return res,vector

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        output = self.output_layer(x)
        return output

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_scores = self.sigmoid(self.linear(x))
        weighted_features = torch.mul(x, attention_scores)
        fused_feature = torch.sum(weighted_features, dim=1, keepdim=True)
        return attention_scores, fused_feature




class Build_MultiModel_szzyy_pair_mlpatt(nn.Module):
    def __init__(self, backbone='resnest50', input_dim=2048, num_classes=2, pretrained_modelpath='None'):
        super().__init__()
        self.num_classes = num_classes
        self.backbone=backbone
        self.model = resnet50()
        self.model.fc = Identity()
        self.lg1 = torch.nn.Linear(in_features=(2048+8), out_features=self.num_classes)
        self.lg2 = torch.nn.Linear(in_features=(2048+2048), out_features=2048)

        self.mlp = MLP(input_dim=(2048+8), hidden_dim=512, output_dim=2)

        self.attention_model = Attention(input_dim=2056)

        self.binary_classifier = nn.Linear(1, 2)


        print('init model:', backbone)


    def forward(self, img, extradata):
        all = []
        output_list = []
        # 循环处理每个图像
        for i in range(len(img)):
            image = img[i]
            temp = self.model(image)
            output_list.append(temp)

        avg_feats = torch.stack(output_list, dim=1)
        summed_feats = torch.sum(avg_feats, dim=1)  # 逐个特征相加
        avg_feats_mean = summed_feats / len(output_list)  # 计算平均特征
        all.append(avg_feats_mean)
        all.append(extradata)
        all_output = torch.cat(all, -1) # b, c1+c2

        if self.backbone=="mlp":

            res = self.mlp(all_output)

        else:
            attention_scores, fused_feature = self.attention_model(all_output)
            res = self.binary_classifier(fused_feature.view(-1, 1))

        return res


 
class Build_MultiModel_szzyy_pair(nn.Module):
    def __init__(self, backbone='resnest50', input_dim=2048, num_classes=2, pretrained_modelpath='None'):
        super().__init__()
        self.num_classes = num_classes
        self.backbone=backbone
        self.model = resnet50()
        self.model.fc = Identity()
        self.lg1 = torch.nn.Linear(in_features=(2048+8), out_features=self.num_classes)
        self.lg2 = torch.nn.Linear(in_features=(2048+2048), out_features=2048)
        print('init model:', backbone)

    def forward(self, img, extradata):
        all = []
        output_list = []
        for i in range(len(img)):
            image = img[i]
            temp = self.model(image)
            output_list.append(temp)


        if self.backbone=="avg":
            avg_feats = torch.stack(output_list, dim=1)
            summed_feats = torch.sum(avg_feats, dim=1)  # 逐个特征相加
            avg_feats_mean = summed_feats / len(output_list)  # 计算平均特征
            all.append(avg_feats_mean)
            all.append(extradata)
            all_output = torch.cat(all, -1) # b, c1+c2
            res = self.lg1(all_output)
        else:
            all_output = torch.cat(output_list, -1) # b, c1+c2
            temp = self.lg2(all_output)
            all.append(temp)
            all.append(extradata)
            all_output = torch.cat(all, -1) # b, c1+c2
            res = self.lg1(all_output)



        return res


class Build_MultiModel_szzyy_pair_multi(nn.Module):
    def __init__(self, backbone='resnest50', input_dim=2048, num_classes=2, pretrained_modelpath='None'):
        super().__init__()
        self.num_classes = num_classes
        self.backbone=backbone
        self.model = resnet50()
        self.model.fc = Identity()
        self.lg1 = torch.nn.Linear(in_features=(2048), out_features=self.num_classes)
        self.lg2 = torch.nn.Linear(in_features=(2048+2048), out_features=2048)
        print('init model:', backbone)

    def forward(self, img, extradata):
        all = []
        output_list = []
        # 循环处理每个图像
        for i in range(len(img)):
            image = img[i]
            temp = self.model(image)
            output_list.append(temp)


        if self.backbone=="avg":
            avg_feats = torch.stack(output_list, dim=1)
            summed_feats = torch.sum(avg_feats, dim=1)  # 逐个特征相加
            avg_feats_mean = summed_feats / len(output_list)  # 计算平均特征

            modified_data = extradata / 10 + 1

            all_output = avg_feats_mean * modified_data



            res = self.lg1(all_output)
        else:
            all_output = torch.cat(output_list, -1) # b, c1+c2
            temp = self.lg2(all_output)
            all.append(temp)
            all.append(extradata)
            all_output = torch.cat(all, -1) # b, c1+c2
            res = self.lg1(all_output)



        return res


class Build_MultiModel_szzyy_pair_onlyimg(nn.Module):
    def __init__(self, backbone='ResNet50',fusion_type='avg', input_dim=2048, num_classes=2, pretrained_modelpath='None'):
        super().__init__()
        self.num_classes = num_classes
        self.backbone=backbone
        self.fusion_type=fusion_type

        if backbone=='ResNet50':
            self.model = resnet50()
            net_dict = self.model.state_dict()
            predict_model = torch.load("/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_standard/resnet50-pre.pth")
            state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}
            net_dict.update(state_dict)
            self.model.load_state_dict(net_dict)

        elif backbone=='ResNet34':
            self.model=resnet34()

        elif backbone=='ResNet50BAM':
            self.model=resnet50bam()
            net_dict = self.model.state_dict()
            predict_model = torch.load("/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_standard/resnet50-pre.pth")
            state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}
            net_dict.update(state_dict)
            self.model.load_state_dict(net_dict)

        elif backbone=='ResNet50CBAM':
            self.model=resnet50cbam()
            net_dict = self.model.state_dict()
            predict_model = torch.load("/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_standard/resnet50-pre.pth")
            state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}
            net_dict.update(state_dict)
            self.model.load_state_dict(net_dict)
        elif backbone=='ResNet50SE':
            self.model=resnet50se()
            net_dict = self.model.state_dict()
            predict_model = torch.load("/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_standard/resnet50-pre.pth")
            state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}
            net_dict.update(state_dict)
            self.model.load_state_dict(net_dict)
        else:
            self.model = resnet152()
        self.model.fc = Identity()

        if self.backbone=='ResNet34':
            self.lg1 = torch.nn.Linear(in_features=(512+512), out_features=self.num_classes)
            self.lg2 = torch.nn.Linear(in_features=(512), out_features=self.num_classes)
        else:
            self.lg1 = torch.nn.Linear(in_features=(2048+2048), out_features=self.num_classes)
            self.lg2 = torch.nn.Linear(in_features=(2048), out_features=self.num_classes)

        print('init model:', backbone)



    def forward(self, img):

        output_list = []
        for i in range(len(img)):
            image = img[i]
            temp = self.model(image)
            output_list.append(temp)
        if self.fusion_type=="avg":
            avg_feats = torch.stack(output_list, dim=1)
            summed_feats = torch.sum(avg_feats, dim=1)  # 逐个特征相加
            avg_feats_mean = summed_feats / len(output_list)  # 计算平均特征
            res = self.lg2(avg_feats_mean)
        else:
            all_output = torch.cat(output_list, -1) # b, c1+c2
            res = self.lg1(all_output)


        return res


class mlpfusion(nn.Module):
    def __init__(self, feature_size = 256): 
        super(mlpfusion, self).__init__()
        self.fc1 = Linear(feature_size*2, 1) 
        self.fc2 = Linear(feature_size*2, 1)
        self.sigmoid= nn.Sigmoid()

    def forward(self, encoder_output_list):
        # pdb.set_trace()
        batch_size = encoder_output_list[0].size()[0]
        xall = torch.cat(encoder_output_list, -1) # b, c1+c2
        weight1 = self.fc1(xall)
        weight2 = self.fc2(xall)
        weight1 = self.sigmoid(weight1)
        weight2 = self.sigmoid(weight2)

        return weight1, weight2


class SegmentConsensus(nn.Module):
    def __init__(self, in_features=2048, out_features=256):
        print("cv fusion: average...", flush=True)
        super(SegmentConsensus, self).__init__()
        self.linear_logits = torch.nn.Linear(
            in_features=in_features, out_features=out_features)

    def forward(self, x):
        x = self.linear_logits(x)
        return x

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
        
# img+img share backbone
class Build_MultiModel_ShareBackbone_mlp(nn.Module):
    def __init__(self,backbone='ResNet34', input_dim=2048, num_classes=1, use_gate=True, pretrained_modelpath='None'):
        super().__init__()
        self.input_dim = input_dim*2
        # self.num_classes = num_classes
        self.use_gate = use_gate
        # self.gate = SegmentConsensus(self.input_dim,2048*5)
        # self.lg1 = torch.nn.Linear(in_features=2048*5, out_features=2048)
        # self.lg2 = nn.Sequential(nn.Linear(256, 2),nn.SoftMax())
        self.lg2 = nn.Sequential(nn.Linear(256, 2), nn.LogSoftmax(dim=1))

        self.backbone=backbone
        if backbone=='ResNet50':
            self.model = resnet50()
            net_dict = self.model.state_dict()
            predict_model = torch.load("/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_standard/resnet50-pre.pth")
            state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}
            net_dict.update(state_dict)
            self.model.load_state_dict(net_dict)

        elif backbone=='ResNet34':
            self.model=resnet34()
        elif backbone=='ResNet50BAM':
            self.model=resnet50bam()
            net_dict = self.model.state_dict()
            predict_model = torch.load("/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_standard/resnet50-pre.pth")
            state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}
            net_dict.update(state_dict)
            self.model.load_state_dict(net_dict)

        elif backbone=='ResNet50CBAM':
            self.model=resnet50cbam()
            net_dict = self.model.state_dict()
            predict_model = torch.load("/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_standard/resnet50-pre.pth")
            state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}
            net_dict.update(state_dict)
            self.model.load_state_dict(net_dict)
        elif backbone=='ResNet50SE':
            self.model=resnet50se()
            net_dict = self.model.state_dict()
            predict_model = torch.load("/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_standard/resnet50-pre.pth")
            state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}
            net_dict.update(state_dict)
            self.model.load_state_dict(net_dict)

        else:
            self.model = resnet152()


        self.model.fc = Identity()

        if self.backbone=='ResNet34':
            self.lgbase = torch.nn.Linear(in_features=512, out_features=256)

        else:
            self.lgbase = torch.nn.Linear(in_features=2048, out_features=256)


        self.mlpfusion=mlpfusion()

        print('init model:', backbone)

        if self.use_gate:
            print('use gate')
        else:
            print('no use gate')


    def forward(self, x):

        encoder_output_list = []
        for i in range(len(x)):
            image = x[i]
            temp = self.model(image)
            temp=self.lgbase(temp)
            encoder_output_list.append(temp)
        weight1, weight2=self.mlpfusion(encoder_output_list)
        encoder_output_list[0] = encoder_output_list[0] * weight1
        encoder_output_list[1] = encoder_output_list[1] * weight2
        avg_feats = torch.stack(encoder_output_list, dim=1)
        summed_feats = torch.sum(avg_feats, dim=1)
        output = summed_feats / image.size(1)
        output = self.lg2(output)

        return output

class Build_MultiModel_szzyy_pair_onlyimg_convnext(nn.Module):
    def __init__(self, backbone='convnext', input_dim=2048, num_classes=2, pretrained_modelpath='None'):
        super().__init__()
        self.num_classes = num_classes
        self.backbone=backbone
        self.model =  create_model(num_classes=num_classes)
        self.model.head = Identity()
        self.lg1 = torch.nn.Linear(in_features=(768+768), out_features=self.num_classes)
        self.lg2 = torch.nn.Linear(in_features=(768), out_features=self.num_classes)
        print('init model:', backbone)

    def forward(self, img):
        output_list = []
        for i in range(len(img)):
            image = img[i]
            temp = self.model(image)
            output_list.append(temp)
        if self.backbone=="avg":
            avg_feats = torch.stack(output_list, dim=1)
            summed_feats = torch.sum(avg_feats, dim=1)  # 逐个特征相加
            avg_feats_mean = summed_feats / len(output_list)  # 计算平均特征
            res = self.lg2(avg_feats_mean)
        else:
            all_output = torch.cat(output_list, -1) # b, c1+c2
            res = self.lg1(all_output)
        return res


# if __name__ == '__main__':
#     model = Build_MultiModel_ShareBackbone()


#     label_path='/root/work2023/deep-learning-for-image-processing-master/data_set/TRSL_ALL'

#     IMAGE_PATH = '/vepfs/gaowh/tr_eyesl/'

#     TRAIN_LISTS = ['train.csv']
#     TEST_LISTS = ['test.csv']
#     val_transforms = Image_Transforms(mode='val', dataloader_id=1, square_size=256, crop_size=224).get_transforms()


#     validate_loader = load_data_multi(label_path, TEST_LISTS, IMAGE_PATH, 16, val_transforms,'test')

#     val_bar = tqdm(validate_loader)
#     for val_data in val_bar:
#         val_images,val_labels,imgids = val_data


#         outputs = model(val_images)
#         print(outputs.shape)

#     # input = torch.randn(4, 40960)
#     # segment_consensus = SegmentConsensus(40960, 256)
#     # output = segment_consensus(input)


#     # inputs = torch.randn(2, 3, 224, 224)
#     # output = model(inputs, inputs)
#     # print(output)


