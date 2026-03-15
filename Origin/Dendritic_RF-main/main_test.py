import shutil
import argparse
import os
import torch
from utils import count_parameters
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## 读取 模型
import time
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from models.VGGSNN import S4Model
import data_loaders     
from functions import TET_loss, seed_all
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('-j',
                    '--workers',
                    default=16,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--epochs',
                    default=200,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=16,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--weight_decay', default=0.05, type=float, help='Weight decay')

parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.001,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('-p',
                    '--print-freq',
                    default=100,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed',
                    default=42,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('--T',
                    default=10,
                    type=int,
                    metavar='N',
                    help='snn simulation time (default: 2)')
parser.add_argument('--means',
                    default=1.0,
                    type=float,
                    metavar='N',
                    help='make all the potential increment around the means (default: 1.0)')
parser.add_argument('--TET',
                    default=True,
                    type=bool,
                    metavar='N',
                    help='if use Temporal Efficient Training (default: True)')
parser.add_argument('--lamb',
                    default=0.0001,
                    type=float,
                    metavar='N',
                    help='adjust the norm factor to avoid outlier (default: 0.0)')
parser.add_argument('--path',
                    default='/data_smr/dataset/cifar10-dvs',
                    type=str,
                    metavar='N',
                    help='path to the dataset')
args = parser.parse_args()


if __name__ == '__main__':
    # seed_all(args.seed)
    train_dataset, val_dataset = data_loaders.build_cifar(use_cifar10=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)

    model = S4Model(d_input=1, d_model=256)
    
    state_dict = torch.load('resnet_256_no_vth.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    
    model = model.cuda()
    print(model)

    import numpy as np
    import seaborn as sns
    Omega = np.linspace(0, 1, 8)
    x_sum = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            images = images.flatten(2)
            images = images.to(device)
            outputs = model(images)

            x_input = model.encoder_bn1(model.encoder1(images))
            x_result_1 = model.layer1.neuron1(x_input)

            print(x_result_1.mean())
            x_result_2 = model.layer1.neuron2(x_input.flip(dims=[-1])).flip(dims=[-1])
            print(x_result_2.mean())

            s = torch.concat([x_result_1, x_result_2], dim=1)
            y = model.layer1.lin1(s.transpose(-1, -2)).transpose(-1, -2)
            x_result = model.layer1.neuron3(y)
            x_sum = x_result.mean()

            print(x_sum.mean())

            x_sum += x_result.mean()
            # print(x_result.shape)
            # print(x_result.mean())
            # break
            x_layer1 = model.bn1(model.layer1(x_result))
            x_result = model.layer2.neuron1(x_layer1)
            # print(x_result.mean())
            x_sum += x_result.mean()

            x_layer2 = model.bn2(model.layer2(x_layer1))
            x_result = model.layer3.neuron1(x_layer2)
            # print(x_result.mean())
            x_sum += x_result.mean()

            x_layer3 = model.bn3(model.layer3(x_layer2))
            x_result = model.layer4.neuron1(x_layer3)
            # print(x_result.mean())
            x_sum += x_result.mean()

            x_layer4 = model.bn4(model.layer4(x_layer3))
            x_result = model.layer5.neuron1(x_layer4)
            # print(x_result.mean())
            x_sum += x_result.mean()

            x_layer5 = model.bn5(model.layer5(x_layer4))
            x_result = model.layer6.neuron1(x_layer5)
            # print(x_result.mean())
            x_sum += x_result.mean()
            print(x_sum / 6)
            break

