# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 20:55:56 2022

@author: Atanas Kolev
"""

import torch
import torch.nn as nn
from torchvision import models
import os

def load_model(last_layer_file_name = 'trained_layer_params.pth', num_categories = 33):
    
    model = models.vgg16(pretrained=True)
    file_path = os.getcwd()+'\\' + last_layer_file_name
    trained_layer = torch.load(file_path)
    num_features = model.classifier[0].out_features
    
    model.classifier = nn.Sequential(*list(model.classifier.children())[:3])
    model.classifier.append(nn.Linear(num_features, num_categories))
    model.classifier[-1].state_dict = trained_layer
    model.classifier[-1].weight.requires_grad = False
    model.classifier[-1].weight[:] = model.classifier[-1].state_dict()['weight']
    model.classifier[-1].bias.requires_grad = False
    model.classifier[-1].bias[:] = model.classifier[-1].state_dict()['bias']
    
    return model