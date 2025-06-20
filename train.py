import torch
import torch.nn as nn
import matplotlib as mat
mat.use('Agg')
import matplotlib.pyplot as plt
import time
import numpy as np
#import get_image as Dataload
import get_image_test as Dataload #加载数据
import ds_ifdm  as db

BATCH = 16
LR = 0.001
EPOCH = 200


def write_file(filename_,data_):
    try:
        with open(filename_,'a+',encoding="utf-8") as fw:
            fw.write(data_)
            fw.write('\n')
            fw.close()
    except Exception as ex:
        print("写入文件失败->error code -1:",ex)
        return -1

# 计算准确率
def calprecise(output, img_gt):
    
    mask = output > 0.5
    acc_mask = torch.mul(mask.float(),img_gt)
    acc_mask = acc_mask.sum()
    acc_fenmu = mask.sum()
    recall_fenmu = img_gt.sum()
 
    p = acc_mask / (acc_fenmu + 0.0001)
    recall = acc_mask / (recall_fenmu + 0.0001)
    f1 = 2*p*recall / (p + recall + 0.0001)
    
    iou = p*recall/ (p+recall- p*recall)
    
 
    return iou, recall, f1


# dice 损失函数
def dice_loss(out, gt, smooth = 1.0):
    gt = gt.view(-1)
    out = out.view(-1)

    intersection = (gt * out).sum()
    dice = (2.0 * intersection + smooth) / (torch.square(gt).sum() + torch.square(out).sum() + smooth) # TODO: need to confirm this matches what the paper says, and also the calculation/result is correct

    return 1.0 - dice


# step为True表示打印每一步的结果
# save_epoch表示多少轮保存一次
def train(print_step=True,  save_epoch=20, loaded_pre = False):
    train = Dataload.NIST_DATA("train")
    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH, shuffle=False)
    # 设置设备信息
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = db.TD_Net_First()
    net = net.to(device)
    if loaded_pre:
        log_dir = './nc16_50_dice.pth'
        checkpoint = torch.load(log_dir)
        net.load_state_dict(torch.load(log_dir))# 加载中模型
        print("Loaded premodel")
    #optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=0.0003)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))
    bce_lossFunction = nn.BCELoss()
    # 训练主流程
    print("开始训练✿✿ヽ(°▽°)ノ✿✿")
    losses = []
    acces = []
    recalles = []
    f1es = []
    for epoch in range(1, EPOCH):
        #if epoch % 2 == 0 & epoch!=0:
        total_loss = 0
        acc_ = 0
        recall_ = 0
        f1es_ = 0
        st = time.time()
        step = 0
        for step,data in enumerate(train_loader):
            img, img_gt, img_hp = data
            img = img.to(device)
            img_hp = img_hp.to(device)
            # img_gt = 1 - img_gt NIST数据集
            img_gt = img_gt.to(device)
            pred_mask = net(img,img_hp)
            pred_mask_sigmoid = pred_mask
            # sigmoid值域是(s0,1)
            pred_mask_flat = pred_mask_sigmoid.view(-1)
            true_masks_flat = img_gt.view(-1)
            #loss = bce_lossFunction(pred_mask_flat, true_masks_flat)
            loss = bce_lossFunction(pred_mask_flat, true_masks_flat)*0.85 + dice_loss(pred_mask_flat, true_masks_flat)*0.15
            step += 1
            # 更新网络
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc, recall, f1 = calprecise(pred_mask, img_gt)
 
            if(print_step == True):
                print("step:%d->loss:%.4f p:%.4f recall:%.4f f1:%.4f cost time:%ds"%
                      (step, loss, acc, recall, f1, time.time()-st))
 
            total_loss = total_loss + loss
            acc_ = acc_ + acc
            recall_ = recall_ + recall
            f1es_ = f1es_ + f1
 
        # 计算每个epoch的平均指标
        loss_mean = total_loss / len(train_loader)
        acc_mean = acc_ / len(train_loader)
        recall_mean = recall_ / len(train_loader)
        f1_mean = f1es_ / len(train_loader)
 
        losses.append(loss_mean)
        acces.append(acc_mean)
        recalles.append(recall_mean)
        f1es.append(f1_mean)
        # 打印这一轮的信息
        ct = time.time() - st
        print("epoch:%d->loss:%.4f p:%.4f recall:%.4f f1:%.4f cost time:%ds" %
              (epoch, loss_mean, acc_mean, recall_mean, f1_mean, ct))
        write_file('./maxblurpool.csv', str(loss_mean)+';'+ str(acc_mean)+';'+str(recall_mean)+';'+str(f1_mean)+';'+str(ct))
        #if epoch % 2 ==0 & epoch!=0:
            #torch.save(net.state_dict(), "./tmp/ds-ifdm-"+str(epoch)+"-dice.pth")
        #time.sleep(10)
    torch.save(net.state_dict(), "./ds-noblur-200-dice.pth")
    
    print("训练结束啦✿✿ヽ(°▽°)ノ✿")
    
    
 
if __name__ == "__main__":
    train()
 