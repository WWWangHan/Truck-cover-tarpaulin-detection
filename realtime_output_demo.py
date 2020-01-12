# -*- coding: utf-8 -*-

import os
import torch
from torch import load, device, cuda, max, sum
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import sys
import time
import logging
import logging.handlers
import finetuning_models as fm

from configparser import ConfigParser

from PIL import Image

def is_valid_image(filename):
    valid = True
    try:
        Image.open(filename).load()
    except OSError:
        valid = False
    return valid

def file_name(file_dir):
    for root, dirs, nameList in os.walk(file_dir):
        pass
    return nameList

### parse config.ini
cf = ConfigParser()
cf.read('cfg/config.ini', encoding= 'utf-8')

# get path from config.ini
photo_path = cf.get('parameter', 'test_photo_path')
model_path = cf.get('parameter', 'test_model_path')
log_path = cf.get('parameter', 'test_log_path')
input_size = int(cf.get('parameter', 'input_size'))

tarp_detect_model_name = 'tarp'
tarp_model_path = model_path + '{}.ckpt'.format(tarp_detect_model_name)

# for network
ip = cf.get('network', 'ip')
port = cf.get('network', 'port')

## logging
formattler = '%(asctime)s   %(message)s'
fmt = logging.Formatter(formattler)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(fmt)

time_handler = logging.handlers.TimedRotatingFileHandler(log_path + 'log.log', when='midnight', interval=1)
time_handler.setLevel(logging.DEBUG)
time_handler.setFormatter(fmt)

logger.addHandler(stream_handler)
logger.addHandler(time_handler)

# model loading
model_tarp, _ = fm.initialize_model('resnet', 4, False, use_pretrained=True)
model_tarp.load_state_dict(torch.load(tarp_model_path))

# run in cpu
device = device('cpu')
model_tarp = model_tarp.to(device)

# data transformer
data_transforms = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# for predicting
model_tarp.eval()

index = 1

while True:

    if os.listdir(photo_path + 'photo/'):
        time.sleep(1)
        image_datasets = datasets.ImageFolder(photo_path, data_transforms)
        # Create training and validation dataloaders
        dataloaders_dict = DataLoader(image_datasets, batch_size=1, shuffle=False)

        with torch.no_grad():
            for blob in dataloaders_dict:

                inputs, _ = blob
                inputs = inputs.to(device)

                outputs_tarp = model_tarp(inputs)
                _, preds_tarp = max(outputs_tarp, 1)
                preds_tarp = preds_tarp.data.cpu().numpy()

                dict_res_tarp = {3: '没盖篷布', 2: '篷布遮盖完好', 1: '部分遮盖', 0: '空车'}

                photo_name = file_name(photo_path)
                license_plate = photo_name[index - 1].split('.')[0]
                log_info = str('车牌号:' + license_plate +
                             '   图片名称: ' + str(photo_name[index - 1]) +
                             '   篷布检测结果: ' + dict_res_tarp[int(preds_tarp)])

                logger.debug(log_info)
                os.remove(photo_path + 'photo/' + photo_name[index - 1])
