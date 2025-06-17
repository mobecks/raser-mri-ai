#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for RASER AI project.
"""

import torch
import numpy as np
import json
from skimage.transform import iradon
from raser_mri_ai.NN_architectures import *
from raser_mri_ai.models.config_models import RaserConfig

# Dummy SimpleNNModel for compatibility (replace with actual implementation if needed)
class SimpleNNModel(torch.nn.Module):
    def __init__(self, input_shape: int, output_shape: int, config: RaserConfig = None):
        super().__init__()
        self.fc = torch.nn.Linear(input_shape, output_shape)
    def forward(self, x: torch.Tensor):
        return self.fc(x)

def create_Sinogram(PATH, data, model_Name: str, angles_Included: bool = True, num_angles: int = 30):
   
    # load weights
    ext_config = open("/home/student/Documents/Raser_AI/outputs/" + model_Name + '/config.json', "r") # load config file
    ext_config = json.loads(ext_config.read()) # read config file
    
    if'FC' in ext_config['arch']:
        model = SimpleNNModel(ext_config['input_shape'],ext_config['output_shape'], config=ext_config)
        model.load_state_dict(torch.load(PATH/"outputs"/  model_Name / "model.pt"))
        model.eval()
    if "TPIfCNN" in ext_config["arch"]:
        
        model = TPIfCNN(input_shape=(int(ext_config["NR_TPIS"]/ext_config['TPI_split']),ext_config['input_shape']), meta_shape=1, out_shape=ext_config['output_shape'], config=ext_config)
        model.load_state_dict(torch.load(PATH/"outputs"/  model_Name / "model.pt"),strict = False)
        model.to(DEVICE)
        model.eval()
    sinogram = np.empty([67,num_angles])
    for i, spectra  in enumerate(data):
        if (angles_Included):
            spectra = np.delete(spectra, [200,201])
        if(spectra.max() == spectra.min()):
            spectra = np.empty(spectra.shape)
        spectra = (spectra - spectra.min())/(spectra.max() - spectra.min())
        spectra = np.append(spectra, 0)
        single_input = torch.from_numpy(spectra)
        single_input = single_input.float()
        single_input = single_input.to(DEVICE)
        single_input = single_input.unsqueeze(0).unsqueeze(1)
        outputs = model(single_input)
        sinogram[:,i] = outputs.cpu().detach().numpy()
    
    return sinogram

def create_Image(PATH, spectra, model_1D, model_2D, angles_Included: bool = True, num_angles: int = 30):
    
    if angles_Included:
        angles = spectra[:,201]
    else:
        lin_space = np.linspace(0,180,30)
        angles = lin_space[0:num_angles]
        
    sinogram = create_Sinogram(PATH,spectra,model_1D, angles_Included = angles_Included, num_angles=num_angles)
    config = open(PATH / "outputs"/  model_2D / 'config.json', "r") # load config file
    config = json.loads(config.read()) # read config file
    if(config["arch"] == "Unet"):
        model = Unet()
        model.load_state_dict(torch.load(PATH/"outputs"/  model_2D / "model.pt"),strict = False)
        model.eval()
        model.to(DEVICE)
    if(config["arch"] == "Autoencoder"):
        model = Autoencoder()
        model.load_state_dict(torch.load(PATH/"outputs"/  model_2D / "model.pt"),strict = True)
        model.eval()
        model.to(DEVICE)
        
    image = iradon(sinogram, theta=angles, output_size = 48,filter_name='cosine')
    in_Img =  iradon(spectra.transpose(), theta=angles, filter_name='cosine')
    image = torch.from_numpy(image)
    image = image.float()
    image = image.to(DEVICE)
    output = model(image.unsqueeze(0).unsqueeze(1))
    
    output = output.squeeze(0).cpu().detach().numpy()
    output = output.squeeze(0)
    
    return output, in_Img

def unpad_Image(image: np.ndarray, in_shape: tuple, out_shape: tuple):

    row_pad = in_shape[0] - out_shape[0]
    col_pad = in_shape[1] - out_shape[1]
    row_start = row_pad // 2
    row_end = -row_pad // 2 or None
    col_start = col_pad // 2
    col_end = -col_pad // 2 or None
    
    
    image = image[row_start:row_end, col_start:col_end]
    return image
