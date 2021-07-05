import os
import sys
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader
import json

import h5py
import re

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET



seq_set = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15] # train  1560
# seq_set = [1, 5, 16] # val  447

xml_dir_list = []
dir_root_gt = 'instruments18_caption/seq_'
annotation = []  

No_caption = []
With_caption = []


for i in seq_set:
    if i == 8:
        continue
    xml_dir_temp = dir_root_gt + str(i) + '/xml_new/'
    xml_dir_list = glob(xml_dir_temp + '/*.xml')   

    random.shuffle(xml_dir_list)  
    total_xml = len(xml_dir_list)  
    print(total_xml)

    for index in range(len(xml_dir_list)):
        file_name = os.path.splitext(os.path.basename(xml_dir_list[index]))[0]
        file_root = os.path.dirname(os.path.dirname(xml_dir_list[index]))
        _xml = ET.parse(xml_dir_list[index]).getroot()
        temp_anno = {}
        tem_fea = {}
          
        if _xml.find('caption') is None:
            id_path = os.path.join("seq_"+str(i),"roi_features_resnet_incremental_ls_cbs/",file_name+"_node_features.npy")
            No_caption.append(id_path)
            continue    
        temp_anno['id_path'] = os.path.join("seq_"+str(i),"roi_features_resnet_incremental_ls_cbs/",file_name+"_node_features.npy")
        temp_anno['caption'] = _xml.find('caption').text
        annotation.append(temp_anno)
        id_path = os.path.join("seq_"+str(i),"roi_features_resnet_incremental_ls_cbs",file_name+"_node_features.npy")
        With_caption.append(id_path)
  

if not os.path.exists('/annotations/annotations_SD_all'):
    os.makedirs('/annotations/annotations_SD_all')

with open('/annotations/annotations_SD_all/captions_val.json', 'w') as f:
    json.dump(annotation, f)


with open('/annotations/annotations_SD_all/WithCaption_id_path_val.json', 'w') as f:
    json.dump(With_caption, f)

print(len(annotation))






   


   

