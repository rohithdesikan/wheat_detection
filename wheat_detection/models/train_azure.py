# %%
# Import base packages
import os
import numpy as np
import pandas as pd
import shutil

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

# Import Azure Packages
from azureml.core.workspace import Workspace
from azureml.core import Experiment

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.train.dnn import PyTorch

# %%
