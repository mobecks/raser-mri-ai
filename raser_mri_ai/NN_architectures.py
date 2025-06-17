#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural network architectures for RASER AI project.
"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from raser_mri_ai.models.config_models import RaserConfig

DEVICE = 'cuda'  # cpu or cuda
#--------------------------------------------1D--------------------------------------------

class TPIfCNN(nn.Module):
    def __init__(self, input_shape=6, meta_shape=1, out_shape=2, config: RaserConfig = None):
        super().__init__()
        self.config = config
        self.spectrum_size = input_shape[1] - meta_shape # Shape of the spectrum
        
        self.kernel_size = config.cnn.kernel_size
        self.filters = config.cnn.filters
        self.stride = config.cnn.stride
        self.dropout_conv = config.cnn.dropout_conv
        self.padding = config.cnn.padding
        self.dilation = config.cnn.dilation
        self.pool_size = config.cnn.pool_size
        self.num_layers_conv = config.cnn.num_layers_conv
        self.smoothing = config.cnn.smoothing
        self.TPI_included = config.cnn.TPI_included
 
        if not self.TPI_included:
            TPI_shape = 0
        else:
            TPI_shape = input_shape[0]
        
        
        
        if config.activation == 'relu': act = nn.ReLU()
        elif config.activation == 'exprelu': act = nn.ELU()
        elif config.activation == 'leakyrelu': act = nn.LeakyReLU()
        elif config.activation == 'sigmoid': act = nn.Sigmoid()
        elif config.activation == 'hardswish': act = nn.Hardswish()
        else: raise ValueError(f"Unknown activation function: {config.activation}")
        
        if config.outlayer == 'relu': actOut = nn.ReLU()
        elif config.outlayer == 'exprelu': actOut = nn.ELU()
        elif config.outlayer == 'leakyrelu': actOut = nn.LeakyReLU()
        elif config.outlayer == 'sigmoid': actOut = nn.Sigmoid()
        elif config.outlayer == 'hardswish': actOut = nn.Hardswish()
        else: raise ValueError(f"Unknown output activation function: {config.outlayer}")
        
        if config.smoothfunction == 'GELU': smooth = nn.GELU()
        elif config.smoothfunction == 'Softplus': smooth = nn.Softplus()
        else: smooth = nn.Identity()
        
        
        
        def one_conv(in_c, out_c, kernel_size, stride, drop_p):
            conv = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=kernel_size, stride=stride),
                nn.ReLU(),
                nn.Dropout(drop_p)
            )
            return conv
                
        block_conv = [] # Holder for all convolutional layers
        block_conv.append(one_conv(1, self.filters, self.kernel_size, self.stride, self.dropout_conv)) # First convolutional layer
        self.outshape = int( (self.spectrum_size + 2*self.padding - self.kernel_size) / self.stride + 1 ) # Calculate the output shape of the convolutional layers
        
        # Append the rest fo the convolutional layers
        for i in range(config.cnn.num_layers_conv - 1):
            block = one_conv(self.filters, self.filters, self.kernel_size, self.stride, self.dropout_conv) # Add a convolutional layer
            block_conv.append(block)
            self.outshape = int( (self.outshape + 2*self.padding - self.kernel_size) / self.stride + 1 ) # Calculate the output shape of the convolutional layers
            if self.pool_size > 1: # Add a pooling layer if needed
                block_conv.append(nn.MaxPool1d(2, stride=self.pool_size))
                self.outshape = int(self.outshape/self.pool_size)
                
        self.i2f = nn.Sequential(*block_conv) # Convert the list of layers to a sequential model

        self.concat_layer = nn.Sequential( # Fully connected layer that combines the convolutional output with the TPI
          nn.Linear(self.outshape*self.filters*input_shape[0] + TPI_shape, config.hidden),
          act,
          nn.BatchNorm1d(num_features=1),
          nn.Dropout(0), # Keep TPI
          )
        
        between_layers = [] # Holder for all fully connected layers
           
        def fc_layer(config): # Define a fully connected layer
            fcl = nn.Sequential(
                nn.Linear(config.hidden, config.hidden),
                act,
                nn.BatchNorm1d(num_features=1),
                nn.Dropout(config.dropout_fc),)  
            return fcl
        
        
        
        for i in range(config.num_layers-1): # Add the fully connected layers
            layer = fc_layer(config)
            between_layers.append(layer)
        self.layers = nn.Sequential(*between_layers)
        
        self.layer_out = nn.Sequential( # Output layer
          nn.Linear(config.hidden, out_shape),
          actOut
        )
        
        self.smooth_layer = nn.Sequential( # Smoothing layer
          nn.Linear(out_shape, out_shape),
          smooth
        )

    def forward(self, inputs):
        
        past_obs = inputs.shape[1] # number of TPI
        
        conv_part = inputs[:,:, :self.spectrum_size]        # spectra
        fc_part = inputs[:,:, self.spectrum_size:]          # TPI
        features = torch.from_numpy(np.zeros([inputs.shape[0], past_obs, self.outshape*self.filters])).float().to(DEVICE) # Holder for the convolutional output
        for k in range(past_obs):                           # convolve each spectrum for its own to keep temporal nature
            features[:,k] = self.i2f(conv_part[:,k].unsqueeze(1)).view(inputs.shape[0],-1) # Run through the convolutional part
        
        if(self.TPI_included): # Add TPI if needed
            combined = torch.cat((features, fc_part), 2) # Add TPI back
        else:
            combined = features
            
        combined = combined.view(combined.shape[0], -1).unsqueeze(1) # Flatten the outputs
        x = self.concat_layer(combined) # Run through the first fc layer
        x = self.layers(x) # Run through the fully connected part
        x = self.layer_out(x)
        if(self.smoothing):
            x = self.smooth_layer(x)
        return(x.squeeze()) # Take away the 2nd dimension and return
    
    
class TPIfComCNN(nn.Module): # CNN which sample the spectra as a whole
    def __init__(self, input_shape=6, meta_shape=1, out_shape=2, config: RaserConfig = None):
        super().__init__()
        self.config = config
        self.spectrum_size = input_shape[1] - meta_shape
        self.kernel_size = config.cnn.kernel_size
        self.filters = config.cnn.filters
        self.stride = config.cnn.stride
        self.dropout_conv = config.cnn.dropout_conv
        self.padding = config.cnn.padding
        self.dilation = config.cnn.dilation
        self.pool_size = config.cnn.pool_size
        self.num_layers_conv = config.cnn.num_layers_conv
        self.smoothing = config.cnn.smoothing
        self.TPI_included = config.cnn.TPI_included
        TPI_shape = input_shape[0] if self.TPI_included else 0

        activations = {
            'relu': nn.ReLU(),
            'exprelu': nn.ELU(),
            'leakyrelu': nn.LeakyReLU(),
            'sigmoid': nn.Sigmoid(),
            'hardswish': nn.Hardswish()
        }
        try:
            act = activations[config.activation]
        except KeyError:
            raise ValueError(f"Unknown activation function: {config.activation}")

        try:
            actOut = activations[config.outlayer]
        except KeyError:
            raise ValueError(f"Unknown output activation function: {config.outlayer}")

        smooth_functions = {
            'GELU': nn.GELU(),
            'Softplus': nn.Softplus()
        }
        smooth = smooth_functions.get(config.smoothfunction, nn.Identity())
        def one_conv(in_c, out_c, kernel_size, stride, drop_p):
            conv = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=kernel_size, stride=stride),
                nn.ReLU(),
                nn.Dropout(drop_p)
            )
            return conv
        block_conv = []
        if self.num_layers_conv > 0:
            block_conv.append(one_conv(input_shape[0], self.filters, self.kernel_size, self.stride, self.dropout_conv))
            self.outshape = int((self.spectrum_size + 2*self.padding - self.kernel_size) / self.stride + 1)
        else:
            self.outshape = self.spectrum_size
            self.filters = input_shape[0]
        for i in range(config.cnn.num_layers_conv - 1):
            block = one_conv(self.filters, self.filters, self.kernel_size, self.stride, self.dropout_conv)
            block_conv.append(block)
            self.outshape = int((self.outshape + 2*self.padding - self.kernel_size) / self.stride + 1)
            if self.pool_size > 1:
                block_conv.append(nn.MaxPool1d(2, stride=self.pool_size))
                self.outshape = int(self.outshape/self.pool_size)
        self.i2f = nn.Sequential(*block_conv)
        self.concat_layer = nn.Sequential(
          nn.Linear(self.outshape*self.filters + TPI_shape, config.hidden),
          act,
          nn.BatchNorm1d(num_features=1),
          nn.Dropout(0),
          )
        between_layers = []
        def fc_layer(config):
            fcl = nn.Sequential(
                nn.Linear(config.hidden, config.hidden),
                act,
                nn.BatchNorm1d(num_features=1),
                nn.Dropout(config.dropout_fc),
            )
            return fcl
        for i in range(config.num_layers - 1):
            layer = fc_layer(config)
            between_layers.append(layer)
        self.layers = nn.Sequential(*between_layers)
        self.layer_out = nn.Sequential(
          nn.Linear(config.hidden, out_shape),
          actOut
        )
        self.smooth_layer = nn.Sequential(
          nn.Linear(out_shape, out_shape),
          smooth
        )

    def forward(self, inputs):
        
        conv_part = inputs[:,:, :self.spectrum_size]        # spectra
        fc_part = inputs[:,:, self.spectrum_size:]          # TPI
        
        if(self.num_layers_conv > 0): # If there are convolutional layers, convolve the spectra
            x = self.i2f(conv_part)
            x = x.view(x.shape[0], -1).unsqueeze(1)
        else: # If there are no convolutional layers, just reshape the spectra
            x = conv_part
            x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]).unsqueeze(1)
        
        all_TPI = fc_part.view(fc_part.shape[0], -1).unsqueeze(1) # Reshape the TPI
        
        if(self.TPI_included):
            x = torch.cat((x, all_TPI), 2) # Add TPI back
        x = self.concat_layer(x) # Run through the first fc layer
        x = self.layers(x) # Run through the fully connected part
        x = self.layer_out(x)
        if(self.smoothing): # Smooth the output
            x = self.smooth_layer(x)
        return(x.squeeze()) # Take away the 2nd dimension and return
        
#--------------------------------------------2D--------------------------------------------
       

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            
            nn.Conv2d(in_channels, out_channels, 3, 1, 1,bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1,bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
               
        )
    def forward(self, x: nn.Module):
        return self.conv(x)


# (mob:) implementation from 
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/model.py
class Unet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, features = [8,16,32,64]):
        super(Unet, self).__init__()
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        #Up_part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                feature*2, feature, kernel_size= 2, stride= 2 
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size = 1)
    
    def forward(self, x: nn.Module):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0,len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape: # om input inte är delbar med 16
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
            
            concant_skip = torch.cat((skip_connection, x), dim = 1)
            x = self.ups[idx+1](concant_skip)
        
        return self.final_conv(x)

class Autoencoder(nn.Module): # Made for 44*44 in image size
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1), # Output shape: (batch_size, 16, 24, 24)
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), # Output shape: (batch_size, 32, 12, 12)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # Output shape: (batch_size, 64, 6, 6)
            nn.ReLU(),
            nn.Flatten(), # Output shape: (batch_size, 2304)
            nn.Linear(in_features=2560, out_features=128), # Output shape: (batch_size, 128)
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=18) # Output shape: (batch_size, 18)
        )

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(in_features=18, out_features=128), # Input shape: (batch_size, 18), Output shape: (batch_size, 128)
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=2560), # Output shape: (batch_size, 2304)
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(64, 10, 4)), # Output shape: (batch_size, 64, 6, 6)
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1), # Output shape: (batch_size, 32, 11, 11)
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1), # Output shape: (batch_size, 16, 22, 22)
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1), # Output shape: (batch_size, 1, 44, 44)
            nn.Sigmoid() # Output shape: (batch_size, 1, 44, 44), values between 0 and 1
        )

    def forward(self, x: nn.Module):
        # Encoding
        x = self.encoder(x)
        # Decoding
        x = self.decoder(x)
        return x

