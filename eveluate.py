import get_image as Dataload
import ds_ifdm as db


import cv2
import time
import numpy as np
import torch
import torch.nn as nn
torch.set_printoptions(profile="full")
import matplotlib
matplotlib.use('Agg') #这样就指定了backend使用了非交互式的，Agg指的是png等 ，也可以指定其它值：PDF,SVG,PS等。
import matplotlib.pyplot as plt

# 测试代码

def write_file(filename_,data_):
    try:
        with open(filename_,'a+',encoding="utf-8") as fw:
            fw.write(data_)
            fw.write('\n')
            fw.close()
    except Exception as ex:
        print("写入文件失败->error code -1:",ex)
        return -1





def calprecise(premask, groundtruth):
    #统计groundtruth中正样本的个数
    GT_pos_sum = np.sum(groundtruth == 1)
    #统计预测的mask中正样本的个数
    Mask_pos_sum = np.sum(premask == 1)
    #统计在groundtruth和mask相同位置都是正样本的个数，即实际为正样本，预测也是正样本的个数
    True_pos_sum = np.sum((groundtruth == 1) * (premask == 1))
    #那么实际为正样本，预测也为正样本占预测的mask中正样本的比例就是Precision
    p = float(True_pos_sum) / (Mask_pos_sum + 1e-6)
    #实际为正样本，预测也为正样本占groundtruth中正样本的比例就是Recall
    r = float(True_pos_sum) / (GT_pos_sum + 1e-6)
    #IoU就是交并比，True_pos_sum就是正样本的交集，groundtruth与premask的正样本相加减去多加了一次的交集，就是最终的交并比
    iou = float(True_pos_sum) / (GT_pos_sum + Mask_pos_sum - True_pos_sum + 1e-6)
    f = 2 * p * r / (p+r + 1e-6)
    return iou,r,f,

     
    
    
    
def eval(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = db.TD_Net_First().to(device)
    net.load_state_dict(torch.load(model))
    val = Dataload.NIST_DATA("train")
    train_loader = torch.utils.data.DataLoader(val, batch_size=1, shuffle=False)
    liou = 0
    lf = 0
    rcl = 0
    ind = 0
    for step, data in enumerate(train_loader):
            img, img_gt, img_hp = data
            img = img.to(device)
            img_hp = img_hp.to(device)
            pred_mask = net(img, img_hp)
            #pred_mask = net(img)
            #print(pred_mask.cpu())
            #print(img_gt)
            mask = pred_mask.cpu()
            #print(pred_mask.cpu())
            save_gt = np.where((img_gt<=0.5),0,255)
            mask = np.where((mask<=0.5),0,255)
            
            #print(type(img_gt)
            cv2.imwrite('./results/renc/step_%d.png'%step, mask[0][0])
            cv2.imwrite('./results/gtnc/step_%d.png'%step, save_gt[0][0])
            pmask = np.where((pred_mask.cpu() <= 0.5),0,1).squeeze()
            gt = np.where((img_gt<=0.5),0,1).squeeze()
            #print(mask.shape)
            print(step)
            #pred_mask = torch.flatten(pred_mask)
            #img_gt = torch.flatten(img_gt)
            #iou, r, f =  accuracy(pred_mask.cpu(), img_gt)
            iou, r, f = calprecise(pmask.reshape(-1), gt.reshape(-1))
            print("step:=> iou:%.4f recall:%.4f f1:%.4f:"%(iou, r, f))
            if f>0:
                ind+=1
                lf+=f
                rcl += r
                liou+=iou
            
            #plt.subplot(1,3,1), plt.imshow(img[0].permute(1,2,0).cpu().numpy()), plt.axis('off')
            #plt.subplot(1,3,2), plt.imshow(img_gt[0][0]), plt.axis('off')
            #plt.subplot(1,3,3), plt.imshow(mask[0][0].detach().numpy()), plt.axis('off')
            #plt.savefig("./res_cov/step_%d"%step)
            #cv2.imwrite("./ps_pics/step_ori_%d.png"%step, img[0].permute(2,1,0).cpu().numpy())
            #cv2.imwrite("./ps_pics/step_pred_%d.png"%step, mask[0][0])
            
            #plt.subplot(1,5,1), plt.imshow(img[0].permute(1,2,0).cpu().numpy())
            #plt.subplot(1,5,3), plt.imshow(img_gt[0][0])
            #plt.subplot(1,5,5), plt.imshow(mask[0][0])
            #plt.savefig("./nc_removal_pics/step_%d"%step)
            #plt.clf()
    #print("iou: ", liou/len(train_loader))
    print("f: ", lf/ind)
    print("r: ", rcl/ind)
    print("iou: ", liou/ind)
    return


 
if __name__ == "__main__":
        model = "dcrru_blur_epoch_102.pth" # 添加已训练好的模型路径
        eval(model)
    #except:
     #   print(1)