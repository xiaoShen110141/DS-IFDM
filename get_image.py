import torch
import torchvision.transforms as tfs
import os
import cv2
import time
import numpy as np
from scipy import ndimage
from torch.nn.parameter import Parameter

# 读取数据集

HEIGHT = 256 #指定高度和宽度
WIDTH = 384

class NIST_DATA(object):
    def __init__(self, mode):
        super(NIST_DATA, self).__init__()
        self.mode = mode 
        self.nist_data_root = "" #要训练/检测的数据的路径
        self.image_name = []
        self.gt_name = []
        self.hp_name = []
        self.readImage()
 
        # 图像正则化参数
        
        self.normMean = [0.45118496, 0.44153717, 0.40016693]
        self.normStd = [0.25564805, 0.25123924, 0.27574804]
        
        self.mean = [0.46730682, 0.4867802, 0.49553254]
        self.std = [0.2466941, 0.23260699, 0.23741737]
        # 原始图像正则化
        self.im_tfs = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize(self.normMean, self.normStd)
        ])

        self.im_tfs_gt = tfs.Compose([tfs.ToTensor()])
        # 高通图像正则化
        self.im_tfs_hp = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize(self.mean, self.std)
        ])
        # self.filter()
        # 训练集配置前80%的图片，测试集配置后20%的图片
        if(mode == "train"):
            self.image_name = self.image_name[:round(len(self.image_name)*1)]
            self.gt_name = self.gt_name[:round(len(self.gt_name)*1)]
            self.hp_name = self.hp_name[:round(len(self.hp_name)*1)]
        
        else:
            self.image_name = self.image_name[round(len(self.image_name) * 0.8):]
            self.gt_name = self.gt_name[round(len(self.gt_name) * 0.8):]
 
        print("%s->一共加载了%d张图片"%(mode,len(self.image_name)))
 


    # 读取图片
    def readImage(self):
        image_root = self.nist_data_root 


        gt_root = '' # groundtruth的路径，可以自己指定
        filename = os.listdir(image_root)
        for i in range(len(filename)):
            self.image_name.append(image_root + filename[i])
            self.gt_name.append(gt_root + filename[i].split(".")[0]+"_mask.png")
            # self.gt_name.append(gt_root + filename[i].replace("t.jpg", "forged.jpg"))
            #self.gt_name.append(gt_root + filename[i].replace('jpg', 'png'))
            #self.hp_name.append(hp_data_root+filename[i].split(".")[0]+"_xy.jpg")
        
 
    # 检查gt和原图是否一致
    def check(self):
        print(len(self.image_name))
        print(len(self.gt_name))
        print(len(self.hp_name))
        #for i in range(0, len(self.image_name)):
            #print("检查: ", self.image_name[i] + " " + self.gt_name[i]+ " " +self.hp_name[i])
    
    # 裁剪图片，默认为256，384，可以自己在HEIGHT那里指定
    def crop(self, img, img_gt, img_hp, height=HEIGHT, width=WIDTH, offset=0):
        #print(img_hp)
        img = cv2.resize(img, (width, height))
        img = self.im_tfs(img)
        img_gt = cv2.resize(img_gt, (width, height))
        img_gt = self.im_tfs_gt(img_gt)
        img_hp = cv2.resize(img_hp, (width, height))
        img_hp = self.im_tfs_gt(img_hp)
        return img, img_gt, img_hp
 
    def image_transform(self, img, img_gt, img_hp):
        
        img, img_gt, img_hp = self.crop(img, img_gt, img_hp)
        # img = self.im_tfs(img)
        # img_gt = self.im_tfs_gt(img_gt)
        # img_hp = self.im_tfs_hp(img_hp)
        return img, img_gt, img_hp

    def __getitem__(self, idx):
        if self.mode == "train":
            #print(self.hp_name[idx])
            #print(self.image_name[idx])
            img = cv2.imread(self.image_name[idx])
            img_gt = cv2.imread(self.gt_name[idx], cv2.IMREAD_GRAYSCALE)
            img_hp = cv2.imread(self.image_name[idx], cv2.IMREAD_GRAYSCALE)
            #print(self.gt_name[idx])
            img, img_gt, img_hp = self.image_transform(img, img_gt, img_hp)
            #print(img_hp)
            return img, img_gt, img_hp


    def __len__(self):
        return len(self.image_name)


if __name__ == "__main__":
    #img = cv2.imread('./mask/NC2016_3436_gt.png')
    #print(img)
    st = time.time()
    train = NIST_DATA("train")
    train.check()
    train_loader = torch.utils.data.DataLoader(train, batch_size=16, shuffle=False)
    
    print("cost time%d"%(time.time()-st))
    for data in train_loader:
        img, img_gt, img_hp = data
        print(img.shape)
        print(img_gt.shape)
        print(img_hp.shape)
        