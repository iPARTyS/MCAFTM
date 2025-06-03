import os
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import random
import time

import numpy as np
from tqdm import tqdm
from medpy.metric import dc, hd95
from scipy.ndimage import zoom

from utils.utils import powerset
from utils.utils import DiceLoss, calculate_dice_percase, val_single_volume
from utils.dataset_ACDC import ACDCdataset, RandomGenerator
from test_ACDC import inference
from networks.networks import MCAFTM
from ptflops import get_model_complexity_info

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=12, help="batch size")  # 12
parser.add_argument("--lr", default=0.0001, help="learning rate")
parser.add_argument("--max_epochs", default=300)
parser.add_argument("--img_size", default=224)
parser.add_argument("--save_path", default="./model_pth/ACDC")
parser.add_argument("--n_gpu", default=1)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--list_dir", default="./data/ACDC/lists_ACDC")
parser.add_argument("--root_dir", default="./data/ACDC/")
parser.add_argument("--volume_path", default="./data/ACDC/test")
parser.add_argument("--z_spacing", default=10)
parser.add_argument("--num_classes", default=4)
parser.add_argument('--test_save_dir', default='./predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')
args = parser.parse_args()

if not args.deterministic:
    cudnn.benchmark = True   # 即使用 CUDA 的优化
    cudnn.deterministic = False  # 表示不使用确定性训练
else:
    cudnn.benchmark = False
    cudnn.deterministic = True

random.seed(args.seed)   # 设置 Python 的随机数生成器的种子为 args.seed，以确保在同一种子下生成的随机数相同。
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

args.is_pretrain = True

args.exp = 'MCAFTM_Small_loss_MUTATION_w3_7_' + str(args.img_size)

snapshot_path = "{}/{}/{}".format(args.save_path, args.exp, 'MCAFTM_Small_loss_MUTATION_w3_7')
snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
snapshot_path = snapshot_path + '_lr' + str(args.lr) if args.lr != 0.01 else snapshot_path
snapshot_path = snapshot_path + '_' + str(args.img_size)
snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

current_time = time.strftime("%H%M%S")
print("The current time is", current_time)
snapshot_path = snapshot_path + '_run' + current_time

# 路径判断，创建
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

args.test_save_dir = os.path.join(snapshot_path, args.test_save_dir)
test_save_path = os.path.join(args.test_save_dir, args.exp)
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path, exist_ok=True)

# 模型初始化 + 加载
net = DDUnet(n_classes=args.num_classes).cuda()

if args.checkpoint:
    net.load_state_dict(torch.load(args.checkpoint))

# 加载数据集
train_dataset = ACDCdataset(args.root_dir, args.list_dir, split="train",
                            transform=transforms.Compose(
                            [RandomGenerator(output_size=[args.img_size, args.img_size])]))

print("The length of train set is: {}".format(len(train_dataset)))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
db_val = ACDCdataset(base_dir=args.root_dir, list_dir=args.list_dir, split="valid")
valloader = DataLoader(db_val, batch_size=1, shuffle=False)  # 用于批量加载验证和测试数据
db_test = ACDCdataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
testloader = DataLoader(db_test, batch_size=1, shuffle=False)

if args.n_gpu > 1:
    net = nn.DataParallel(net)

net = net.cuda()
net.train()


ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(args.num_classes)
# save_interval = args.n_skip

iterator = tqdm(range(0, args.max_epochs), ncols=70)   # 迭代次数
iter_num = 0

Loss = []
Test_Accuracy = []

Best_dcs = 0.80
Best_dcs_th = 0.865



logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

max_iterations = args.max_epochs * len(train_loader)  # 记录训练的迭代次数
base_lr = args.lr  # 初始学习率


optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=0.0001)



def val():
    logging.info("Validation ===>")
    dc_sum = 0
    metric_list = 0.0
    net.eval()
    for i, val_sampled_batch in enumerate(valloader):
        val_image_batch, val_label_batch = (val_sampled_batch["image"],
                                            val_sampled_batch["label"])

        val_image_batch, val_label_batch = (val_image_batch.squeeze(0).cpu().detach().numpy(),
                                            val_label_batch.squeeze(0).cpu().detach().numpy())

        x, y = val_image_batch.shape[0], val_image_batch.shape[1]
        if x != args.img_size or y != args.img_size:
            val_image_batch = zoom(val_image_batch, (args.img_size / x, args.img_size / y),
                                   order=3)  # not for double_maxvits
        val_image_batch = torch.from_numpy(val_image_batch).unsqueeze(0).unsqueeze(0).float().cuda()


        outputs1, outputs2, masks, stage_out1, _ = net(val_image_batch, [])
        val_outputs = outputs1 + outputs2
        val_outputs = torch.softmax(val_outputs, dim=1)
        val_outputs = torch.argmax(val_outputs, dim=1).squeeze(0)
        val_outputs = val_outputs.cpu().detach().numpy()
        if x != args.img_size or y != args.img_size:
            val_outputs = zoom(val_outputs, (x / args.img_size, y / args.img_size), order=0)
        else:
            val_outputs = val_outputs

        dc_sum += dc(val_outputs, val_label_batch[:])
    performance = dc_sum / len(valloader)
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, Best_dcs))

    print('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, Best_dcs))
    # print("val avg_dsc: %f" % (performance))
    return performance




for epoch in iterator:
    net.train()
    train_loss = 0
    # 遍历
    for i_batch, sampled_batch in enumerate(train_loader):
        image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
        image_batch, label_batch = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor)
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

        lc1, lc2 = 0.3, 0.7
        for num in range(3):
            if num == 0:
                outputs1, outputs2, masks, stage_out1, _ = net(image_batch, [])
            else:
                outputs1, outputs2, masks, stage_out1, _ = net(image_batch, en)
            en = []
            for idx in range(len(masks[0])):
                mask1 = masks[0][idx].detach()
                mask2 = masks[1][idx].detach()
                en.append(1e-3 * (mask1 - mask2))
            out5, out4, out3, out2, out1 = stage_out1[0], stage_out1[1], stage_out1[2], stage_out1[3], stage_out1[4]
            loss = 0.0
            # 计算每个层次的交叉熵损失和 Dice 损失，并加权求和，得到总体损失。
            loss5 = lc1 * ce_loss(out5, label_batch[:].long()) + lc2 * dice_loss(out5, label_batch, softmax=True)
            loss4 = lc1 * ce_loss(out4, label_batch[:].long()) + lc2 * dice_loss(out4, label_batch, softmax=True)
            loss3 = lc1 * ce_loss(out3, label_batch[:].long()) + lc2 * dice_loss(out3, label_batch, softmax=True)
            loss2 = lc1 * ce_loss(out2, label_batch[:].long()) + lc2 * dice_loss(out2, label_batch, softmax=True)
            loss1 = lc1 * ce_loss(out1, label_batch[:].long()) + lc2 * dice_loss(out1, label_batch, softmax=True)

            loss_outputs1 = lc1 * ce_loss(outputs1, label_batch[:].long()) + lc2 * dice_loss(outputs1, label_batch,
                                                                                             softmax=True)
            loss_outputs2 = lc1 * ce_loss(outputs2, label_batch[:].long()) + lc2 * dice_loss(outputs2, label_batch,
                                                                                             softmax=True)

            loss += (loss_outputs1 + loss_outputs2 + loss1 + loss2 + loss3 + loss4 + loss5)

            optimizer.zero_grad()  # 清零优化器梯度。
            loss.backward()  #  反向传播损失。
            optimizer.step() #  更新模型参数。

            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 # We did not use this
            lr_ = base_lr
            # 优化器
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

        iter_num = iter_num + 1
        if iter_num % 50 == 0:
            logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
            print('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
        train_loss += loss.item()
    Loss.append(train_loss / len(train_dataset))   # 相当于没执行一个epoch记录一下损失值
    logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
    print('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))

    save_model_path = os.path.join(snapshot_path, 'last.pth')
    torch.save(net.state_dict(), save_model_path)

    avg_dcs, mean_hd95, mean_jacard, mean_asd = inference(args, net, testloader, args.test_save_dir)
    if (avg_dcs > Best_dcs):
        Best_dcs = avg_dcs
        save_mode_path = os.path.join(snapshot_path, 'best.pth')
        torch.save(net.state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))
        print("save model to {}".format(save_model_path))

    # if avg_dcs > Best_dcs_th or avg_dcs >= Best_dcs:
    #     avg_test_dcs, avg_hd, avg_jacard, avg_asd = inference(args, net, testloader, args.test_save_dir)
    #     print("test avg_dsc: %f" % (avg_test_dcs))
    #     logging.info("test avg_dsc: %f" % (avg_test_dcs))
    #     Test_Accuracy.append(avg_test_dcs)

    if epoch >= args.max_epochs - 1:
        save_model_path = os.path.join(snapshot_path, 'epoch={}_lr={}_avg_dcs={}.pth'.format(epoch, lr_, avg_dcs))
        torch.save(net.state_dict(), save_model_path)
        logging.info("save model to {}".format(save_model_path))
        print("save model to {}".format(save_model_path))
        iterator.close()
        break
