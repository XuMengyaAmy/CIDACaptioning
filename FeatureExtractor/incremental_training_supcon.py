'''
Paper : Incremental Domain Adaptation with Smoothing and Calibration for Surgical Report Generation
'''

import os
import copy
import time
import random
import argparse

import numpy as np
import PIL.Image as Image

import torch


import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn.functional import binary_cross_entropy

from model.resnet import ResNet18
from model import resnet_cbs

from model.celosswithls import CELossWithLS
from model.instrument_class_dataset import SurgicalClassDataset18_incremental_transform, memory_managment

import warnings
warnings.filterwarnings("ignore")


'''============== For supcon ================'''
from model.resnet_big import SupConResNet
from model.resnet_big_cbs import SupConResNet_cbs

from losses import SupConLoss
from SupCon_util import TwoCropTransform

import matplotlib.pyplot as plt
import pylab


def seed_everything(seed=12):
    '''
    seed randoms for all libraries
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def train_supcon(args, period, net, net_old, train_loader, criterion, optimizer):

    num_exp = 0
    loss_avg = 0
    tstart = time.clock()

    # set net to train mode
    net.train()
    net_old.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = torch.cat([data[0], data[1]], dim=0) 

        # send data amd target to cuda
        data = data.cuda()
        target = target.cuda()

        bsz = target.shape[0] 

        #compute loss
        features = net(data)  
        
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) 

        if args.method == 'SupCon':
            loss_normal = criterion(features, target)
        elif args.method == 'SimCLR':
            loss_normal = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(args.method))
        
        ''' ---------------------- Distilation loss based on old net --------------------------'''
        if (period > 0):
            with torch.no_grad():
                # old network output
                features_old = net_old(data)
                
            f1_old, f2_old =  torch.split(features_old, [bsz, bsz], dim=0)
            features_old_new_f1 = torch.cat([f1_old.unsqueeze(1), f1.unsqueeze(1)], dim=1)
            features_old_new_f2 = torch.cat([f2_old.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            if args.method == 'SupCon':
                loss_dist = (criterion(features_old_new_f1, target) + criterion(features_old_new_f2, target))/2
        else: loss_dist = 0.0
        ''' ------------------ End of distillation loss based on old net ---------------------- '''       
        
        # loss calculatoin
        loss = loss_normal + args.dist_ratio*loss_dist
   

        loss_avg += loss.item()
        num_exp += np.shape(target)[0]

        loss.backward()
        optimizer.step()

    # average calculation
    loss_avg /= num_exp

    # time calculation
    tend = time.clock()
    tcost = tend - tstart

    # return(tcost, loss_avg, acc_avg)
    return(tcost, loss_avg)


def set_model_supcon(args):

    #model = SupConResNet(name=args.model)
    ############################################
    if args.cbs  == 'True':
        print("SupConResNet_cbs")
        model = SupConResNet_cbs(name=args.model, args=args)
    else:
        print("SupConResNet")
        model = SupConResNet(name=args.model)

    criterion = SupConLoss(temperature=args.temp)

    
    if torch.cuda.is_available():
        print('GPU number = %d' % (torch.cuda.device_count()))
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
       
    else: print('only cpu is available')

    return model, criterion

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

    '''--------------------------------------------------- Arguments ------------------------------------------------------------'''
    parser = argparse.ArgumentParser(description='Incremental learning for feature extraction')

    parser.add_argument('--epoch_base',         type=int,       default = 30,                   help='number of base epochs') 
    parser.add_argument('--epoch_finetune',     type=int,       default = 15,                   help='number of finetune epochs') 
    parser.add_argument('--batchsize',          type=int,       default = 20,                   help='batchsize')
    parser.add_argument('--stop_acc',           type=float,     default = 0.998,                help='number of epochs') 
 
    parser.add_argument('--lr',                 type=float,     default = 0.001,                help='learning rate') 
    parser.add_argument('--gamma',              type=float,     default = 0.8,                  help='learning gamma')
    parser.add_argument('--ft_lr_factor',       type=float,     default = 0.1,                  help='learning rate')  
    parser.add_argument('--momentum',           type=float,     default = 0.6,                  help='learning momentum') 
    parser.add_argument('--decay',              type=float,     default = 0.0001,               help='learning rate')
    parser.add_argument('--dist_ratio',         type=float,     default = 0.5,                  help='learning scheduler')
    parser.add_argument('--schedule_interval',  type=int,       default = 5,                    help='learning rate')
    parser.add_argument('--T',                  type=float,     default = 3.0,                  help='learning rate')
    
    parser.add_argument('--period_train',       type=int,       default = 2,                    help='number of periods')  
    parser.add_argument('--num_class',          type=int,       default = 11,                   help='number of classes ')
    parser.add_argument('--num_class_novel',                    default = [0,9,11],             help='number of novel classes') 
    parser.add_argument('--memory_size',                        default = 50,                   help='number of classes ') 

    parser.add_argument('--ls',                 type=str,       default = 'False',              help='use label smoothing')
    # CBS ARGS
    parser.add_argument('--cbs',                type=str,       default = 'False',              help='use cbs')
    parser.add_argument('--std',                type=float,     default=1,                      help='The initial standard deviation value') 
    parser.add_argument('--std_factor',         type=float,     default=0.9,                    help='curriculum learning: decrease the standard deviation of the Gaussian filters with the std_factor')
    parser.add_argument('--epoch_decay',        type=int,       default=2,                      help='decay the standard deviation value every 5 epochs') 
    parser.add_argument('--kernel_size',        type=int,       default=3,                      help='kernel_size')
    
    # SupCon ARGS
    parser.add_argument('--save_freq',   type=int, default=10, help='save frequency')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='surgical',help='dataset')
    parser.add_argument('--size', type=int, default=224, help='parameter for RandomResizedCrop')   
    # method
    parser.add_argument('--method', type=str, default='SupCon', choices=['SupCon', 'SimCLR'], help='choose method')
    # temperature
    parser.add_argument('--temp', type=float, default=0.07, help='temperature for loss function')  
    
    args = parser.parse_args()
    '''-------------------------------------------------------------------------------------------------------------------------'''
   
    print(args)
    print('ls',args.ls)
    print('cbs',args.cbs)

    # seed the randoms
    seed_everything()

    '''============================= For supcon ===================================='''
    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(size=args.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ])  



    memory_train = [] 
    
    # learning rate schedules
    schedules = range(args.schedule_interval, args.epoch_base, args.schedule_interval)

    class_order = np.arange(args.num_class) # class_order = np.random.permutation(args.num_class)
    
    print('class order:', class_order)

    # get model state
    model_path = 'checkpoint_SupCon/first_model_e2e_aug_%d_%s%s' % (0, ''.join(str(e) for e in class_order[args.num_class_novel[0]:args.num_class_novel[1]]), '.pkl')
    flag_model = os.path.exists(model_path)

    net, criterion_supcon = set_model_supcon(args)  

    # initializing classes and accuracy
    class_old = np.array([], dtype=int)

    for period in range(args.period_train):

        print('===================== period = %d ========================='%(period))

        # current 10 classes
        class_novel = class_order[args.num_class_novel[period]:args.num_class_novel[period+1]]
        print('class_novel:', class_novel)

        ''' ==================== dataloader for stage1(SupCon): Encoder + Projector  ========================= '''
        # combined train dataloader
        combined_train_dataset = SurgicalClassDataset18_incremental_transform(classes=list(range(args.num_class_novel[period],args.num_class_novel[period+1])), memory=memory_train, transform=TwoCropTransform(train_transform), is_train=True) #seq_set=seq_set_train, TwoCropTransform=True, 
        combined_train_loader = DataLoader(dataset=combined_train_dataset, batch_size= args.batchsize, shuffle=True, num_workers=2, drop_last=False)
        print('train size: ', len(combined_train_loader.dataset), ' , Data: new samples:', list(range(args.num_class_novel[period],args.num_class_novel[period+1])), '+ memory:', len(memory_train))

        ''' copy net-old for knowledge distillation '''
        net_old = copy.deepcopy(net)
        '''----------------------------------------'''       

        # initialize variables
        lrc = args.lr
        acc_training = []
        print('current lr = %f' % (lrc))

        # epoch training
        for epoch in range(args.epoch_base):


            if args.cbs  == 'True':
                ###################### cbs: pass the epoch into "get_new_kernels" to update the std  #####################
                net.encoder.module.get_new_kernels(epoch)
                net_old.encoder.module.get_new_kernels(epoch)
            
            net.cuda()
            net_old.cuda()
          
            # load pretrained model
            if period == 0 and flag_model:
                print('load model: %s' % model_path)
                # net.load_state_dict(torch.load(model_path))
                net.load_state_dict(torch.load(model_path)['state_dict'])

            '''===================================  training combined  ==========================================''' 
            # decaying learning rate
            if epoch in schedules:
                lrc *= args.gamma
                print('current lr = %f' % (lrc))

            # update lr in optimizer
            optimizer = torch.optim.SGD(net.parameters(), lr=lrc, momentum=args.momentum, weight_decay=args.decay)
            # optimizer = torch.optim.Adam(net.parameters(), lr=lrc, betas=(0.9, 0.98))

            '''======== Stage 1  ============'''
            tcost, loss_avg = train_supcon(args, period, net, net_old, combined_train_loader, criterion_supcon, optimizer) 
            print('Training Period: %d \t Epoch: %d \t time = %.1f \t loss = %.6f' % (period, epoch, tcost, loss_avg))
            
            ##################### Save model ##########################################
            if (epoch+1) % args.save_freq == 0:
                model_path = 'checkpoint_SupCon/base_model_e2e_aug_%d_%s%s' % (0, ''.join(str(e) for e in class_order[args.num_class_novel[0]:args.num_class_novel[period+1]]), '.pkl')
                if not os.path.isdir('checkpoint_SupCon/'):
                    os.makedirs('checkpoint_SupCon/')
                print('save model: %s' % model_path)
                torch.save({
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'period':period,
                    'epoch': epoch,
                    'args': args
                    }, model_path)
            '''================================== End training combined  ========================================='''


        ''' copy net-old for finetuning '''
        net_old = copy.deepcopy(net)
        '''----------------------------------------'''

        '''========================================  Finetuning ===================================================='''
        # finetune
        if period > 0:

            # initialize variables
            acc_finetune_train = []
            lrc = args.lr*args.ft_lr_factor # small learning rate for finetune
            print('finetune current lr = %f' % (lrc))

            for epoch in range(args.epoch_finetune):

                # fine tune train_dataloaders
                finetune_combined_train_dataset = SurgicalClassDataset18_incremental_transform(classes=list(range(args.num_class_novel[period],args.num_class_novel[period+1])), memory=memory_train, fine_tune_size = ((args.num_class_novel[period+1]-args.num_class_novel[period])*args.memory_size), transform=TwoCropTransform(train_transform), is_train=True,) #seq_set=seq_set_train, TwoCropTransform=True, 
                finetune_combined_train_loader = DataLoader(dataset=finetune_combined_train_dataset, batch_size= args.batchsize, shuffle=True, num_workers=2, drop_last=False)
                if(epoch == 0):  print('finetune train size:', len(finetune_combined_train_loader.dataset))

                '''===================================  training combined  ==========================================''' 
                # learning rate
                if epoch in schedules:
                    lrc *= args.gamma
                    print('current lr = %f'%(lrc))

                # criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(net.parameters(), lr=lrc, momentum=args.momentum, weight_decay=args.decay)
                # optimizer = torch.optim.Adam(net.parameters(), lr=lrc, betas=(0.9, 0.98))
                
                # finetune the model based on binary cross entropy loss and distillation loss
                tcost, loss_avg = train_supcon(args, period, net, net_old, finetune_combined_train_loader, criterion_supcon, optimizer) #, finetune=True
                print('Finetune Training Period: %d \t Epoch: %d \t time = %.1f \t loss = %.6f'%(period, epoch, tcost, loss_avg))
        
                ##################### Save model ##########################################
                if epoch % args.save_freq == 0:
                    model_path = 'checkpoint_SupCon/finetune_model_e2e_aug_%d_%s%s' % (0, ''.join(str(e) for e in class_order[args.num_class_novel[0]:args.num_class_novel[period+1]]), '.pkl')
                    print('save model: %s' % model_path)
                    torch.save({
                        'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'period':period,
                        'epoch': epoch,
                        'args': args
                        }, model_path)
                '''================================== End training combined  ========================================='''


        # save the last model at the end of each period
        model_path = 'checkpoint_SupCon/first_model_e2e_aug_%d_%s%s' % (0, ''.join(str(e) for e in class_order[args.num_class_novel[0]:args.num_class_novel[period+1]]), '.pkl')
        print('save model: %s' % model_path)
        torch.save({
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'period':period,
            'epoch': epoch,
            'args': args
            }, model_path)

        '''-------------------------------------------- random images selection -------------------------------------------'''
        new_added_memory = memory_managment(classes=list(range(args.num_class_novel[period],args.num_class_novel[period+1])), fine_tune_size=((args.num_class_novel[period+1]-args.num_class_novel[period])*args.memory_size)) #seq_set=seq_set_train, 
        memory_train = memory_train + new_added_memory  # memory0;   memory0 + memory1
        print('memory_train:', len(memory_train))
        # append new class images 
        class_old = np.append(class_old, class_novel, axis=0)






   
    







