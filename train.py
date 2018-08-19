import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

import numpy as np

vgg16 = models.vgg16_bn(pretrained=True)
