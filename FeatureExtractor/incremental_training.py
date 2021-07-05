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
from model.instrument_class_dataset import SurgicalClassDataset18_incremental, memory_managment

import warnings
warnings.filterwarnings("ignore")

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


def train (args, period, net, net_old, train_loader, loss_criterion, loss_activation, optimizer, class_old, class_novel, finetune):
    '''
    arguments: period, net, net_old, train_loader, loss_activation, optimizer, clss_old, clasS_novel, finetune
    returns: tcost, loss_avg, acc_avg
    '''

    acc_avg = 0
    num_exp = 0
    loss_avg = 0
    loss_cls_avg = 0
    loss_dist_avg = 0
    tstart = time.clock()

    # set net to train mode
    net.train()
    net_old.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        
        # loaded model, do not need training
        if period == 0 and flag_model and not finetune:
            num_exp = 1
            break # exit the train

        optimizer.zero_grad()

        # prepare target_onehot
        bs = np.shape(target)[0]
        target_onehot = np.zeros(shape = (bs, args.num_class), dtype=np.int) 
        for i in range(bs): target_onehot[i,target[i]] = 1
        target_onehot = torch.from_numpy(target_onehot)
        target_onehot = target_onehot.float()

        # send data and target to cuda
        data = data.cuda()
        target_onehot = target_onehot.cuda()

        # predict output
        output = net(data) # args.num_class
    
        # indices for combined classes
        class_indices = torch.LongTensor(np.concatenate((class_old, class_novel), axis=0))
        class_indices = class_indices.cuda()

        # loss for network
        output_new_onehot = torch.index_select(output, 1, class_indices) 
        target_onehot = torch.index_select(target_onehot, 1, class_indices) 
        
        combined_loss = loss_criterion(output_new_onehot, target_onehot)

        ''' ---------------------- Distilation loss based on old net --------------------------'''
        if (period > 0):

            if not finetune:
                # indices for old classes
                class_indices = torch.LongTensor(class_old)
                class_indices = class_indices.cuda()
                    
            # current_network output
            dist = torch.index_select(output, 1, class_indices)
            dist = dist/args.T

            with torch.no_grad():
                # old network output
                output_old = net_old(data)
                output_old = torch.index_select(output_old, 1, class_indices)
            target_dist = Variable(output_old)#, requires_grad=False)  
            target_dist = target_dist/args.T
            #loss_dist = loss_criterion(dist, loss_activation(target_dist))
            loss_dist = F.binary_cross_entropy(loss_activation(dist), loss_activation(target_dist))
            
        else: loss_dist = 0.0
        ''' ------------------ End of distillation loss based on old net ---------------------- '''

        # loss calculatoin
        loss = combined_loss + args.dist_ratio*loss_dist
        loss_avg += loss.item()
        loss_cls_avg += combined_loss.item()
        if period == 0: loss_dist_avg += 0
        else:loss_dist_avg += loss_dist.item()

        acc = np.sum(np.equal(np.argmax(output_new_onehot.cpu().data.numpy(), axis=-1), np.argmax(target_onehot.cpu().data.numpy(), axis=-1)))
        acc_avg += acc
        num_exp += np.shape(target)[0]

        #optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # average calculation
    loss_avg /= num_exp
    loss_cls_avg /= num_exp
    loss_dist_avg /= num_exp
        
    # average calculation
    acc_avg /= num_exp

    # time calculation
    tend = time.clock()
    tcost = tend - tstart

    return(tcost, loss_avg, acc_avg)


def test(args, net, test_loader, loss_activation, class_old, class_novel):
    '''
    arguments: net, test_loader, loss_activation, class_old, class_novel
    return: tcost, acc_avg
    '''

    acc_avg = 0
    num_exp = 0
    tstart = time.clock()

    # set net to eval
    net.eval()
    net_old.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):

            # prepare target_onehot
            bs = np.shape(target)[0]
            target_onehot = np.zeros(shape = (bs, args.num_class), dtype=np.int)
            for i in range(bs): target_onehot[i,target[i]] = 1
            target_onehot = torch.from_numpy(target_onehot)
            target_onehot = target_onehot.float()

            # send image and target to cuda
            data = data.cuda()
            target_onehot = target_onehot.cuda()

            # predict output
            output = net(data)  

            # indices for combined classes
            class_indices = torch.LongTensor(np.concatenate((class_old, class_novel), axis=0))
            class_indices = class_indices.cuda() 

            # calculate output and target one_hot
            output = torch.index_select(output, 1, class_indices)  
            output = loss_activation(output)
            output = output.cpu().data.numpy()
            target_onehot = torch.index_select(target_onehot, 1, class_indices) 
            #target_onehot = target_onehot[:, np.concatenate((class_old, class_novel), axis=0)]

            # calculation accuracy
            acc = np.sum(np.equal(np.argmax(output, axis=-1), np.argmax(target_onehot.cpu().data.numpy(), axis=-1)))
            acc_avg += acc
            num_exp += np.shape(target)[0]

    # calculate average accuracy
    acc_avg /= num_exp
            
    # time calculation
    tend = time.clock()
    tcost = tend - tstart

    return(tcost, acc_avg)

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
    parser.add_argument('--num_class_novel',                    default = [0,9,11],            help='number of novel classes') 
    parser.add_argument('--memory_size',                        default = 50,                  help='number of classes ') 

    parser.add_argument('--ls',                 type=str,       default = 'False',               help='use label smoothing')
    # CBS ARGS
    parser.add_argument('--cbs',                type=str,       default = 'False',               help='use cbs')
    parser.add_argument('--std',                type=float,     default=1,                      help='The initial standard deviation value') 
    parser.add_argument('--std_factor',         type=float,     default=0.9,                    help='curriculum learning: decrease the standard deviation of the Gaussian filters with the std_factor')
    parser.add_argument('--epoch_decay',        type=int,       default=2,                      help='decay the standard deviation value every 5 epochs')  
    parser.add_argument('--kernel_size',        type=int,       default=3,                      help='kernel_size')
    
    
    args = parser.parse_args()
    '''-------------------------------------------------------------------------------------------------------------------------'''
   
    print(args)
    print('ls',args.ls)
    print('cbs',args.cbs)

    # seed the randoms
    seed_everything()

    memory_train = [] 
    
    # learning rate schedules
    schedules = range(args.schedule_interval, args.epoch_base, args.schedule_interval)

    class_order = np.arange(args.num_class) # class_order = np.random.permutation(args.num_class) 
    print('class order:', class_order)

    # get model state
    model_path = 'checkpoint/first_model_e2e_aug_%d_%s%s' % (0, ''.join(str(e) for e in class_order[args.num_class_novel[0]:args.num_class_novel[1]]), '.pkl')
    flag_model = os.path.exists(model_path)

    # network, loss
    ############   ResNet with cbs   ###################
    if args.cbs  == 'True':
        net = resnet_cbs.ResNet18(args)
        print('==================')
        print("ResNet18 with CBS")
    else:
        net = ResNet18(args.num_class)
        print('==================')
        print("ResNet18")

    loss_activation = nn.Softmax(dim=1).cuda()

    if args.ls  == 'True': 
        print('smoothing = 0.1')
        loss_criterion = CELossWithLS(smoothing = 0.1, gamma=0.0, isCos=False, ignore_index=-1).cuda()
    else:
        print('smoothing = 0.0')
        loss_criterion = CELossWithLS(smoothing= 0.0, gamma=0.0, isCos=False, ignore_index=-1).cuda()

    # gpu
    num_gpu = torch.cuda.device_count()
    if num_gpu > 0:
        print('GPU number = %d' % (num_gpu))
        device_ids = np.arange(num_gpu).tolist()
        print('device_ids:', device_ids)
        net = nn.DataParallel(net, device_ids=device_ids).cuda()
    else: print('only cpu is available')
    
    
    # initializing classes and accuracy
    class_old = np.array([], dtype=int)                # old class
    acc_nvld_basic = np.zeros((args.period_train))     # accuracy list
    acc_nvld_finetune = np.zeros((args.period_train))  # accuracy list

    for period in range(args.period_train):

        '''-----------------------------------------------------'''
        '''
        For period>0, best_accuracy is the best one considering two stages(base stage and fine-tune stage). 
        best_accuracy_finetune is the best one from fine-tune stage.
        
        if best_accuracy = best_accuracy_finetune (exactly same epoch), the best model is from fine-tune stage. This means fine-tune improve the test performance.
        if best_accuracy != best_accuracy_finetune, the best model is from base stage. This means fine-tune does not improve the test performance.
        '''
        # save the best model with best test accuracy for each period
        best_accuracy = 0
        best_epoch = 0

        # saving best model just for fine tune stage
        best_accuracy_finetune = 0
        best_epoch_finetune = 0

        print('===================== period = %d ========================='%(period))

        # current 10 classes
        class_novel = class_order[args.num_class_novel[period]:args.num_class_novel[period+1]]  
        print('class_novel:', class_novel)

        # combined train dataloader
        combined_train_dataset = SurgicalClassDataset18_incremental(classes=list(range(args.num_class_novel[period],args.num_class_novel[period+1])), memory=memory_train, is_train=True) 
        combined_train_loader = DataLoader(dataset=combined_train_dataset, batch_size= args.batchsize, shuffle=True, num_workers=2, drop_last=False)
        print('train size: ', len(combined_train_loader.dataset), ' , Data: new samples:', list(range(args.num_class_novel[period],args.num_class_novel[period+1])), '+ memory:', len(memory_train))
        # test dataloader
        test_dataset = SurgicalClassDataset18_incremental(classes=list(range(0,args.num_class_novel[period+1])), is_train=False) 
        test_loader = DataLoader(dataset=test_dataset, batch_size= args.batchsize, shuffle=True, num_workers=2, drop_last=False)
        print('test size: ', len(test_loader.dataset), ' , Data: samples:', list(range(0,args.num_class_novel[period+1])))
        
        print('Length of dataset- train:', combined_train_dataset.__len__(), ' valid:', test_dataset.__len__())

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
                net.module.get_new_kernels(epoch)
                net_old.module.get_new_kernels(epoch)
            
            net.cuda()
            net_old.cuda()


            # load pretrained model
            if period == 0 and flag_model:
                print('load model: %s' % model_path)
                net.load_state_dict(torch.load(model_path))

            '''===================================  training combined  ==========================================''' 
            # decaying learning rate
            if epoch in schedules:
                lrc *= args.gamma
                print('current lr = %f' % (lrc))

            # update lr in optimizer
            optimizer = torch.optim.SGD(net.parameters(), lr=lrc, momentum=args.momentum, weight_decay=args.decay)

            # train model based on binary cross entropy and distillation loss
            tcost, loss_avg, acc_avg = train(args, period, net, net_old, combined_train_loader, loss_criterion, loss_activation, optimizer, class_old, class_novel, False)

            # log accuracy
            acc_training.append(acc_avg)

            print('Training Period: %d \t Epoch: %d \t time = %.1f \t loss = %.6f \t acc = %.4f' % (period, epoch, tcost, loss_avg, acc_avg))
            '''================================== End training combined  ========================================='''

            '''===================================== Test combined  =============================================='''
            # test model
            tcost, test_acc_avg = test(args, net, test_loader, loss_activation, class_old, class_novel)

            # log test accuracy
            acc_nvld_basic[period] = test_acc_avg

            print('Test(n&o)Period: %d \t Epoch: %d \t time = %.1f \t\t\t\t acc = %.4f' % (period, epoch, tcost, test_acc_avg))

            ################### save model based on the best test accuracy ################################
            if test_acc_avg > best_accuracy:
                best_accuracy = test_acc_avg
                best_epoch = epoch

                model_path = 'checkpoint/best_model_e2e_aug_%d_%s%s' % (0, ''.join(str(e) for e in class_order[args.num_class_novel[0]:args.num_class_novel[period+1]]), '.pkl')
                print('save best model in base stage: %s' % model_path)
                torch.save({
                    'state_dict': net.state_dict(),
                    'period':period,
                    'epoch': epoch,
                    'test_accuracy':test_acc_avg,
                    'args': args
                    }, model_path)
            ###############################################################################################

            # if loaded model, exit the epoch test
            if period == 0 and flag_model: break 

            if len(acc_training)>20 and acc_training[-1]>args.stop_acc and acc_training[-5]>args.stop_acc:
                print('training loss converged')
                break
            '''==============================  End of test combined  =============================================='''

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
                finetune_combined_train_dataset = SurgicalClassDataset18_incremental(classes=list(range(args.num_class_novel[period],args.num_class_novel[period+1])), memory=memory_train, fine_tune_size = ((args.num_class_novel[period+1]-args.num_class_novel[period])*args.memory_size), is_train=True) #transform=train_transform,  seq_set=seq_set_train,  
                finetune_combined_train_loader = DataLoader(dataset=finetune_combined_train_dataset, batch_size= args.batchsize, shuffle=True, num_workers=2, drop_last=False)
                if(epoch == 0):  print('finetune train size:', len(finetune_combined_train_loader.dataset))

                '''===================================  training combined  ==========================================''' 
                # learning rate
                if epoch in schedules:
                    lrc *= args.gamma
                    print('current lr = %f'%(lrc))

                # criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(net.parameters(), lr=lrc, momentum=args.momentum, weight_decay=args.decay)

                # finetune the model based on binary cross entropy loss and distillation loss
                tcost, loss_avg, acc_avg = train(args, period, net, net_old, finetune_combined_train_loader, loss_criterion, loss_activation, optimizer, class_old, class_novel, True) # finetune_combined_train_loader

                # log fine tune accuracy
                acc_finetune_train.append(acc_avg)

                print('Finetune Training Period: %d \t Epoch: %d \t time = %.1f \t loss = %.6f \t acc = %.4f'%(period, epoch, tcost, loss_avg, acc_avg))
                '''================================== End training combined  ========================================='''

                '''===================================== Test combined  =============================================='''
                # test model
                tcost, test_acc_avg = test(args, net, test_loader, loss_activation, class_old, class_novel)

                # test accuracy list
                acc_nvld_finetune[period] = test_acc_avg

                print('Finetune Test(n&o) Period: %d \t Epoch: %d \t time = %.1f \t\t\t\t acc = %.4f' % (period, epoch, tcost, test_acc_avg))

                ################### save model based on the best test accuracy ################################
                if test_acc_avg > best_accuracy:
                    best_accuracy = test_acc_avg
                    best_epoch = epoch

                    model_path = 'checkpoint/best_model_e2e_aug_%d_%s%s' % (0, ''.join(str(e) for e in class_order[args.num_class_novel[0]:args.num_class_novel[period+1]]), '.pkl')
                    print('save best model in fine tune stage: %s' % model_path)
                    torch.save({
                        'state_dict': net.state_dict(),
                        'period':period,
                        'epoch': epoch,
                        'test_accuracy':test_acc_avg,
                        'args': args
                        }, model_path)

                if test_acc_avg > best_accuracy_finetune:
                    best_accuracy_finetune = test_acc_avg
                    best_epoch_finetune = epoch

                    model_path = 'checkpoint/best_model_finetune_e2e_aug_%d_%s%s' % (0, ''.join(str(e) for e in class_order[args.num_class_novel[0]:args.num_class_novel[period+1]]), '.pkl')
                    print('saving best model just for fine tune stage: %s' % model_path)
                    torch.save({
                        'state_dict': net.state_dict(),
                        'period':period,
                        'epoch': epoch,
                        'test_accuracy':test_acc_avg,
                        'args': args
                        }, model_path)
                ###############################################################################################
                
            
                if len(acc_finetune_train) > 20 and acc_finetune_train[-1] > args.stop_acc and acc_finetune_train[-5] > args.stop_acc:
                    print('training loss converged')
                    break
        '''===================================  End of test combined  ================================================='''

        if period>0:
            print('------------------- result ------------------------')
            print('Period: %d, basic acc = %.4f, finetune acc = %.4f' % (period, acc_nvld_basic[period], acc_nvld_finetune[period]))
            print('---------------------------------------------------')

        if period == args.period_train-1:
            print('------------------- ave result ------------------------')
            print('basic acc = %.4f, finetune acc = %.4f' % (np.mean(acc_nvld_basic[1:], keepdims=False), np.mean(acc_nvld_finetune[1:], keepdims=False)))
            print('---------------------------------------------------')

        print('===========================================================')

        # save the last model at the end of each period
        model_path = 'checkpoint/first_model_e2e_aug_%d_%s%s' % (0, ''.join(str(e) for e in class_order[args.num_class_novel[0]:args.num_class_novel[period+1]]), '.pkl')
        print('save model: %s' % model_path)
        torch.save(net.state_dict(), model_path)

        '''-------------------------------------------- random images selection -------------------------------------------'''
        new_added_memory = memory_managment(classes=list(range(args.num_class_novel[period],args.num_class_novel[period+1])), fine_tune_size=((args.num_class_novel[period+1]-args.num_class_novel[period])*args.memory_size))#seq_set=seq_set_train, 
        memory_train = memory_train + new_added_memory  # memory0;   memory0 + memory1
        print('memory_train:', len(memory_train))
        #append new class images 
        class_old = np.append(class_old, class_novel, axis=0)
    
    # accuracy list
    print('acc_base    : ', acc_nvld_basic)    # just show the test accuracy from the last epoch during the base training
    print('acc_finetune: ', acc_nvld_finetune) # just show the test accuracy from the last epoch during the finetuning training
    
    print('===========================================================')
    print('The best model from the finetune stage is')
    print('acc_best_test_finetune: ', best_accuracy_finetune)
    print('epoch_best_finetune   : ', best_epoch_finetune)
    
    print('------------------------------------------------------------')
    print('The best model from the all stages (base + finetune) is')
    print('acc_best_test: ', best_accuracy)
    print('epoch_best   : ', best_epoch)

    print('------------------------------------------------------------')

    best_model_path = 'checkpoint/best_model_e2e_aug_%d_%s%s' % (0, ''.join(str(e) for e in class_order[args.num_class_novel[0]:args.num_class_novel[period+1]]), '.pkl')
    data = torch.load(best_model_path)
    net.load_state_dict(data['state_dict'])
    print('period = %d,  best epoch %d, best accuracy %f' % (data['period'], data['epoch'], data['test_accuracy']))







