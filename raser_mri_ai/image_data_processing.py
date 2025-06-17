#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data processing for RASER AI project.
"""

# %% Imports
#import nmrglue as ng  # nmrglue-0.9.dev0
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from raser_mri_ai.NN_architectures import *
from raser_mri_ai.models.config_models import RaserConfig


RAYTUNE = False  # HPO
DEVICE = 'cuda'  # cpu or cuda


# Hpyerparameters
initial_config = RaserConfig(
    id=10,
    set='Soeren',
    os='Linux',
    Dataset='CTPhantom',
    sim_rounds=1,
    TPIs=1,
    TPI_included=1,
    input_shape=202,
    output_shape=67,
    signal_shape=4096,
    angle_count=30,
    img_size=44,
    x2D=True
    # cnn=... if needed
)

# Grabs the path based on the os: Linux returns /home/student
# Windows returns 'C\\Users\\Username'
if (initial_config.os == 'Linux'):
     PATH = os.path.join(os.path.expanduser('~'), 'Documents', 'Raser_AI')
     DATAPATH = PATH + '/Data/Datasets/'  + initial_config.Dataset + '/'
     processed_Datapath = PATH + '/Data/Processedata/' + initial_config.Dataset
     # Training and testing
     fnames_A0 = glob.glob(DATAPATH + '**/**//**//A(0).csv')
     fnames_d0 = glob.glob(DATAPATH + '**/**//**//d(0).csv')
     fnames_Phi0 = glob.glob(DATAPATH + '**/**//**//Phi(0).csv')
     fnames_out = glob.glob(DATAPATH + '**/**//**//output*.csv')
     fnames_meta = glob.glob(DATAPATH + '**/**//**//meta.csv')
     folders = glob.glob(DATAPATH + '**')

elif (initial_config.os == 'Windows'):
    PATH = os.path.join(os.path.expanduser('~'), 'Documents', 'Github')
    PATH = os.path.join(os.path.expanduser('~'), 'Documents', 'Projekt', 'TQME33')
    DATAPATH = PATH + '\\Data\\Datasets\\' + initial_config.Dataset + '\\'
    processed_Datapath = PATH + '\\Data\\Processedata\\' + initial_config.Dataset

    folders = glob.glob(DATAPATH + '**\\**\\')
else:
    raise ValueError(f"Unknown OS specified in initial_config.os: {initial_config.os}")
    

# Sort on last value


## Inputs
# Check if a output folder exist with the current id, if it doesnt a new folder is created
if not os.path.exists(processed_Datapath) :
    os.makedirs(processed_Datapath)


sim_rounds = initial_config.sim_rounds  # 10 for data, 2 for data2, 1 for when data is really large
if(initial_config.TPI_included):
    initial_config.input_shape = 202
    data_all = np.empty([len(folders), initial_config.angle_count,
                        initial_config.input_shape]-1)
else:
    data_all = np.empty([len(folders), initial_config.TPIs,
                        initial_config.input_shape])

angle_all = np.empty([len(folders), initial_config.angle_count,initial_config.TPIs])


## Targets
labels_all = np.empty(
    [len(folders), initial_config.angle_count, initial_config.output_shape])

images_all = np.empty([len(folders), initial_config.img_size, initial_config.img_size])



# necessary? remove overhead nans
data_all[:] = np.nan
labels_all[:] = np.nan
angle_all[:] = np.nan
images_all[:] = np.nan


# Testing set prep
badIndexes = []
print(folders[0])

#%% Read the files
all_signals = np.empty([ len(folders), initial_config['signal_shape'], 2])
# Sort the folders
folders.sort(key=lambda fname: int(fname.split('e')[-1]))

for idx_f in range(len(folders)):
    try:
        # get all the possible angles
        
        anglePaths = glob.glob(folders[idx_f] + '/**')
        # delete the images
        if(initial_config['2D']):
            image_path = list(filter(lambda x:'png' in x, anglePaths))
            images_all[idx_f] = plt.imread(image_path[0])
        
        anglePaths = list(filter(lambda x: not 'png' in x, anglePaths))
        #Sort in angle order 1,2...19,20
        anglePaths.sort(key=lambda fname: int(fname.split('/')[-1]))
      #  print(folders[idx_f])
        
        for idx_a in range(len(anglePaths)):
            signal_paths = glob.glob(anglePaths[idx_a] + '/**/output*.csv')
            angle_paths = glob.glob(anglePaths[idx_a] + '/**/meta.csv')
            target_paths = glob.glob(anglePaths[idx_a] + '/**/d(0).csv') 
            
            signals = np.empty([ initial_config['TPIs'], initial_config['signal_shape'], 2])
            angles = np.empty([initial_config['TPIs'], 1])
            test = 0
           
           
            for idx_p in range(len(signal_paths)):
                
                #angles in the end of the input
                angles[idx_p,:] = np.genfromtxt(angle_paths[idx_p])[1]
                 
                
                # Signal part
                inputs_tmp = np.genfromtxt(signal_paths[idx_p])
                inputs_tmp = np.swapaxes(inputs_tmp, 0, 1)
                nan_indices = ~np.isnan(inputs_tmp).any(axis=1)
                inputs_tmp = inputs_tmp[nan_indices]
                
                inputs_complex = np.empty([int(inputs_tmp.shape[0]/2), inputs_tmp.shape[1], 2])
                inputs_complex[:, :, 0] = inputs_tmp[::2, :]
                inputs_complex[:, :, 1] = inputs_tmp[1::2, :]
                inputs = inputs_complex
                
                # Noise?
                
                signals[idx_p] = inputs
                if(idx_p == 0):
                
                    all_signals[idx_f] = signals[0]
                  
                
            for i, d in enumerate(signals):
                compl = np.pad(d[:, 0], (2*len(d), 2*len(d)), 'constant', constant_values=np.mean(d[-10:, 0])) \
                                    + np.pad(d[:, 1], (2*len(d), 2*len(d)), 'constant',
                                              constant_values=np.mean(d[-10:, 1])).astype(complex)
    
                # abs of fft
                spec = np.abs(np.fft.fft(compl))[:int(len(compl)/2)]
                spec[0] = 0  # first point is offset frq.
    
                pos = np.argmax(spec)
                width = 200
                out = spec[int(pos-width/2):int(pos+width/2)]
    
                # append pos and scale
                d = np.append(out, pos/2048)
                
                # Made now to only work for one RASER spectra
                data_all[idx_f, idx_a] = d
            
                angle_all[idx_f, idx_a, 0] = angles
                
                target = np.genfromtxt(target_paths[i])
                labels_all[idx_f, idx_a] = (target-target.min())/(target.max()-target.min()) # Local normalization
    except:
        badIndexes.append(idx_f)
    


#################################################
# TODO
# create batched input OR reshape
# Join all TPI:s with the correct spectra
if(initial_config['TPI_included']):
    # input size is now 202
    data_all = np.concatenate((data_all, angle_all), axis=2)  
    
# Remove any nan values
data_all = np.delete(data_all, badIndexes, axis=0)
labels_all = np.delete(labels_all, badIndexes, axis=0)
images_all = np.delete(images_all, badIndexes, axis=0)

    
    

#%%

# flatten
#data_all = data_all.reshape(-1, data_all.shape[-1]) Don't 
#labels_all = labels_all.reshape(-1, labels_all.shape[-1])


if(initial_config['os'] == 'Linux'):
    np.save(processed_Datapath + '/'+ 'data' , data_all)
    np.save(processed_Datapath + '/'+ 'labels',labels_all)
    if(initial_config['2D']):
        np.save(processed_Datapath + '/' + 'images', images_all)
    #np.save(processed_Datapath + '/'+ 'angels',Angels_all)
else:
    np.save(processed_Datapath + '\\'+ 'data' , data_all)
    np.save(processed_Datapath + '\\'+ 'labels',labels_all)
    if(initial_config['2D']):
        np.save(processed_Datapath + '\\' + 'images', images_all)

def concatenate_datasets(name_S1: str, name_S2: str) -> None:

    if(initial_config['os'] == 'Linux'):
        data_Path1 = PATH + '/Data/Processedata/' + name_S1 
        data_Path2 = PATH + '/Data/Processedata/' + name_S2
    else:
        data_Path1 = PATH + '\\Data\\Processedata\\' + name_S1 
        data_Path2 = PATH + '\\Data\\Processedata\\' + name_S2
    
    data_One = np.load(data_Path1 + '/data.npy')
    labels_One = np.load(data_Path1 + '/labels.npy')
    image_One = np.load(data_Path1 + '/images.npy')
    data_Two = np.load(data_Path2 + '/data.npy')
    labels_Two = np.load(data_Path2 + '/labels.npy')
    image_Two = np.load(data_Path2 + '/images.npy')

    data_Comb = np.concatenate((data_One,data_Two), axis = 0)
    labels_Comb = np.concatenate((labels_One,labels_Two), axis = 0)
    images_Comb = np.concatenate((image_One,image_Two), axis = 0)
    
    
    
    if not os.path.exists(data_Path1 + 'comb') :
        os.makedirs(data_Path1 + 'comb')
        
    
    np.save(data_Path1 + 'comb' + '/'+ 'data' , data_Comb)
    np.save(data_Path1 + 'comb' + '/'+ 'labels',labels_Comb)
    np.save(data_Path1 + 'comb' + '/'+ 'images',images_Comb)
#concatenate_datasets('1kImagesWithPump', '1153_images')

