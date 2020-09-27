# %%
# Import base packages
import os
import numpy as np
import pandas as pd

# Import viz packages
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Import PyTorch packages
from sklearn.model_selection import train_test_split
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
def get_data(traindir):
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
class Wheat_Dataset(Dataset):
    def __init__(self, traindir, bbox_data):
        self.traindir = traindir
        self.bbox_data = bbox_data
        self.filenames = os.listdir(self.traindir)
    
    # Get 1 image to train on
    def __get_item__(self, index):

        # Get the image path, open the indexed image and convert to Tensor
        image_id = self.filenames[index]
        image_path = os.path.join(self.traindir, image_id)
        image_raw = Image.open(image_path)
        image_arr = np.array(image_raw).reshape((3, 1024, 1024))
        image = torch.Tensor(image_arr)

        image_name = index.split('.')[0]
        boxes = torch.Tensor(self.bbox_data[image_name])
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        
        targets = {'boxes': boxes, 'labels': labels}

        return image, targets
    
    # Return the length of these filenames
    def __len__(self):
        return len(self.filenames)


# How to collate the files within each batch. This gets loaded into the train and test loaders
def collate_fn(batch):
    return tuple(zip(*batch))

# %%
# Set up the Pytorch model in a FasterRCNN class and set all parameters to be trainable
class FasterRCNN(nn.Module):
    def __init__(self):
        super(FasterRCNN, self).__init__()
        self.frcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.frcnn_model.roi_heads.box_predictor.cls_score.in_features
        self.frcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 1)

    def forward(self, images, targets):
        x = self.frcnn_model(images, targets)

        return x

# %%
# Get the train data loader after performing the necessary preprocessing from the utils file
def _get_train_data_loader(resize):
    logger.info("Get train data loader")

    images_path = os.path.join(args.train_dir, 'image')
    targets_path = os.path.join(args.train_dir, 'annos')
    file_ids = [file_id.split('.')[0] for file_id in os.listdir(images_path)]

    # Load the dataset, pass it through the preprocessing steps from the utils file and load the data loader
    dataset = TransformData(images_path, targets_path, file_ids, resize=resize)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn = collate_fn)

    return train_loader

