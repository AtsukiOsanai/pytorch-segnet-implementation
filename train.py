import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

from models.segnet import segnet

vgg16bn = models.vgg16_bn(pretrained=True)
model = segnet()
model.init_vgg16bn_params(vgg16bn)
if touch.cuda.is_available():
    model.cuda()

optimizer = optim.SGD(model.parameters, lr=0.1, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

def train(args):
    for epoch in range(args.n_epoch):
        # Training
        model.train()
        for i, (images, labels) in enumerate(trainloader):
            images = Variable(images)
            labels = Variable(labels)
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        for i, (images, labels) in enumerate(validloader):
            images = Variable(images)
            labels = Variable(labels)
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            results = model(images)
            loss = loss_fn(results, labels)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--arch', nargs='?', type=str, default='segnet',
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--n_epoch', nargs='?', type=int, default='1e+4',
                        help='The number of epochs')
    args = parser.parse_args()
    train(args)
