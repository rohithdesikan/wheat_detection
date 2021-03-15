# Import base packages
import numpy as np
import argparse
import json
import logging
import os
import sys
import csv
import ast

from PIL import Image

# Import PyTorch packages
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.optim as optim

# %%
# Set up image folder locations
currdir = os.getcwd()
parentdir = os.path.abspath(os.path.join(currdir, os.pardir))
datadir = os.path.join(parentdir, 'data')
rawdir = os.path.join(datadir, 'raw')
traindir = os.path.join(rawdir, 'train')

# %%
def get_data():
    data = {}
    for img_file in os.listdir(traindir):
        img_name = img_file.split('.')[0]
        data[img_name] = data.get(img_name, [])


    with open(os.path.join(rawdir, 'train.csv')) as file:
        reader = csv.DictReader(file)
        for row in reader:
            data[row['image_id']].append(ast.literal_eval(row['bbox']))
    
    return data

# %%
# Set up the PyTorch transforms step

