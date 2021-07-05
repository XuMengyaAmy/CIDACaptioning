'''
Paper: Class-Incremental Domain Adaptation with Smoothing and Calibration for Surgical Report Generation
'''
# System
import numpy as np
import sys
import os
import cv2
from PIL import Image
from glob import glob

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import torchvision.models
import torch

from PIL import Image
import math
import pickle

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
import torch.nn.functional as F

# from resnet import ResNet18
from model.resnet import ResNet18
from model import resnet_cbs

from collections import OrderedDict

import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ACTION_CLASSES = (
    'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation', 'Tool_Manipulation', 'Cutting', 'Cauterization'
    , 'Suction', 'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing')

mlist = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16]
dir_root_gt = '/instruments18_caption/seq_'

xml_dir_list = []

for i in mlist:
    xml_dir_temp = dir_root_gt + str(i) + '/xml_new/' 
    seq_list_each = glob(xml_dir_temp + '/*.xml')
    xml_dir_list = xml_dir_list + seq_list_each

'''==================== change the network based on whether there is CBS ====================='''
parser = argparse.ArgumentParser(description='Arguments of CBS version')

parser.add_argument('--num_class',          type=int,       default = 11,                   help='number of classes ')
# CBS ARGS
parser.add_argument('--cbs',                type=str,       default = 'False',               help='use cbs')
parser.add_argument('--std',                type=float,     default=1,                      help='The initial standard deviation value') 
parser.add_argument('--std_factor',         type=float,     default=0.9,                    help='curriculum learning: decrease the standard deviation of the Gaussian filters with the std_factor')
parser.add_argument('--epoch_decay',        type=int,       default=5,                      help='decay the standard deviation value every 5 epochs') 
parser.add_argument('--kernel_size',        type=int,       default=3,                      help='kernel_size')


parser.add_argument('--model', type=str, default='resnet18')
args = parser.parse_args()

############### For ResNet without CBS ########################
# feature_network = ResNet18(args.num_class).cuda()

############### For ResNet with CBS ########################
# feature_network = resnet_cbs.ResNet18(args).cuda()
# feature_network.get_new_kernels(0) # need initialize the get_new_kernels function
# feature_network.cuda()
##############################################################

############### For SupCon ########################################################
from model.resnet_big import SupConResNet, LinearClassifier
from model.resnet_big_cbs import SupConResNet_cbs, LinearClassifier_cbs
########################### For no cbs #############################################
# model = SupConResNet(name=args.model).cuda()
# classifier = LinearClassifier(name=args.model, num_classes=args.num_class).cuda()
# classifier = torch.nn.DataParallel(classifier)
######################### For cbs ################################
model = SupConResNet_cbs(name=args.model, args=args).cuda()
# classifier = LinearClassifier_cbs(name=args.model, num_classes=args.num_class).cuda()
model.encoder.get_new_kernels(0) 
model.cuda()
# classifier = torch.nn.DataParallel(classifier)
###################################################################

model_dir = 'pretrained_feature_extractor/first_model_e2e_aug_0_012345678910.pkl' 
feature_folder = 'roi_features_resnet18_inc_sup_cbs'

ckpt = torch.load(model_dir, map_location='cpu')
state_dict = ckpt['state_dict']

# Load parameters for encoder
if torch.cuda.is_available():
    print('GPU number = %d' % (torch.cuda.device_count()))
    if torch.cuda.device_count() > 1:
        model.encoder = torch.nn.DataParallel(model.encoder)
    else:
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
else: print('only cpu is available')

classifier = nn.Flatten()
classifier = classifier.cuda()

model.eval()
classifier.eval()

for index, _xml_dir in enumerate(xml_dir_list):
    img_name = os.path.basename(xml_dir_list[index][:-4])
    _img_dir = os.path.dirname(os.path.dirname(xml_dir_list[index])) + '/left_frames/' + img_name + '.png'
    save_data_path = os.path.join(os.path.dirname(os.path.dirname(xml_dir_list[index])), feature_folder)
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)
    _img = Image.open(_img_dir).convert('RGB')
    _xml = ET.parse(_xml_dir).getroot()

    det_classes = []
    act_classes = []
    node_bbox = []
    det_boxes_all = []
    c_flag = False
    print(_img_dir)
    for obj in _xml.iter('objects'):  
        name = obj.find('name').text.strip()
        bbox = obj.find('bndbox')
        interact = obj.find('interaction').text.strip()
        act_classes.append(ACTION_CLASSES.index(str(interact)))
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []


        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text)
            bndbox.append(cur_pt)
        node_bbox += [bndbox]
        det_boxes_all.append(np.array(bndbox))

    node_num = len(act_classes)
    instrument_num = node_num - 1
    adj_mat = np.zeros((node_num, node_num))
    adj_mat[0, :] = act_classes
    adj_mat[:, 0] = act_classes
    adj_mat = adj_mat.astype(int)
    adj_mat[adj_mat > 0] = 1

    node_labels = np.zeros((node_num, len(ACTION_CLASSES)))
    for edge_idx in range(node_num):
        if act_classes[edge_idx] > 0:
            node_labels[0, act_classes[edge_idx]] = 1
            node_labels[edge_idx, act_classes[edge_idx]] = 1
            bndbox = np.hstack((np.minimum(node_bbox[0][:2], node_bbox[edge_idx][:2]),
                                np.maximum(node_bbox[0][2:], node_bbox[edge_idx][2:])))

            det_boxes_all.append(bndbox)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    input_h, input_w = 224, 224

    # roi features extraction
    feature_size = 512
    node_features = np.zeros((np.array(node_bbox).shape[0], feature_size))
    edge_features = np.zeros((node_num, node_num, feature_size))
    roi_idx = 0
    adj_idx = np.where(adj_mat[0, :] == 1)[0]
    edge_idx = 0
    _img = np.array(_img)
    for bndbox in det_boxes_all:
        roi = np.array(bndbox).astype(int)
        roi_image = _img[roi[1]:roi[3] + 1, roi[0]:roi[2] + 1, :]
        roi_image = np.asarray(roi_image, np.float32) / 255
        roi_image = cv2.resize(roi_image, (input_h, input_w), interpolation=cv2.INTER_LINEAR)
        roi_image = torch.from_numpy(np.array(roi_image).transpose(2, 0, 1, )).float()
        roi_image = torch.autograd.Variable(roi_image.unsqueeze(0)).cuda()
        
        ######################### switch between the two lines #####################################
        # feature = feature_network(roi_image) #  for not supcon one 
        feature = classifier(model.encoder(roi_image)) # for supcon one 
        #######################################################################################
        feature = feature.squeeze(0)

        if roi_idx < node_num:
            node_features[roi_idx, ...] = feature.data.cpu().numpy()
        else:
            edge_features[0, adj_idx[edge_idx]] = feature.data.cpu().numpy()
            edge_features[adj_idx[edge_idx], 0] = feature.data.cpu().numpy()
            edge_idx += 1
        roi_idx += 1

    np.save(os.path.join(save_data_path, '{}_edge_features'.format(img_name)), edge_features)
    np.save(os.path.join(save_data_path, '{}_node_features'.format(img_name)), node_features)
print('Done')
