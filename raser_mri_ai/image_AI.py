#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image AI processing for RASER AI project.
This script handles image data loading, model training, evaluation, and visualization for the RASER MRI AI project.
"""

#%% Data processing
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as T
import ray
import matplotlib.pyplot as plt
from skimage.transform import iradon
from skimage.metrics import structural_similarity as ssim, mean_squared_error as image_mse
from pathlib import Path
from raser_mri_ai.NN_architectures import *
from raser_mri_ai.utility_functions import *
from raser_mri_ai.models.config_models import RaserConfig

DEVICE = 'cuda' 
RAYTUNE = False  # Set to True to enable Ray Tune hyperparameter search
# Configuration dictionary for experiment setup
initial_config = RaserConfig(
    id="21kImg_2",
    set='Soeren',
    os='Linux',
    Dataset='100_images',
    Image_Size=44,
    arch='Unet',
    input='Iradon',
    output='image',
    Testingset='CTPhantom',
    Ext_Model='Linux_TPIfCNN_21kimg',
    Testing=0,
    Scheduler=False,
    TPI_included=False,
    TPI_split=16,
    LR=4.145e-05,
    WD=1.013e-06,
    epochs=1,
    batch_size=1,
    input_shape=201,
    output_shape=67,
    loss='MAE',
    optimizer='adam',
)


# PATH to the RASER_AI folder
PATH = Path("/home/student/Documents/Raser_AI")

# Load external model configuration
ext_config = open(PATH / "outputs" / initial_config.Ext_Model / 'config.json', "r")
ext_config = json.loads(ext_config.read())




# angle_all = np.load(PATH + '/Data/Processedata/' + 'Imgangle' +'/angels.npy')
# angle_all_id = ray.put(angle_all)
# angle = ray.get(angle_all_id)
# angle = angle.reshape(30)

# angle = np.append(angle, [25,30])
angle = np.linspace(0, 180, 48)
model_String = initial_config.os +'_' + initial_config.arch + '_' + str(initial_config.id)
OUTPATH = PATH / "outputs" / model_String
if not os.path.exists(OUTPATH):
    os.makedirs(OUTPATH)

#Training


data_all = np.load(PATH /'Data/Processedata' / initial_config.Dataset /'data.npy')
labels_all = np.load(PATH / 'Data/Processedata' / initial_config.Dataset /'labels.npy')
train_images = np.load(PATH / 'Data/Processedata' / initial_config.Dataset / 'images.npy')
train_images_id = ray.put(train_images)

#Testing
input_all = np.load(PATH / 'Data/Processedata' / initial_config.Testingset / 'data.npy')
test_all = np.load(PATH / 'Data/Processedata' / initial_config.Testingset / 'labels.npy')
test_images = np.load(PATH / 'Data/Processedata' / initial_config.Testingset / 'images.npy')
test_images_id = ray.put(test_images)

# send to ray
data_all_id = ray.put(data_all)
image_all_id = ray.put(train_images)
input_all_id = ray.put(input_all)

# Assign labels_all_id and test_all_id based on input/output configuration
if(initial_config.input == "Iradon" and initial_config.output == "image"):
    labels_all_id = ray.put(train_images)
    test_all_id = ray.put(test_images)
elif(initial_config.input == "Sinogram" and initial_config.output == "Sinogram"):
    labels_all_id = ray.put(labels_all)
    test_all_id = ray.put(test_all)
elif(initial_config.input == "Sinogram" and initial_config.output == "image"):
    labels_all_id = ray.put(train_images)
    test_all_id = ray.put(test_images)
else:
    labels_all_id = ray.put(labels_all)
    test_all_id = ray.put(test_all)

# Function that creates the train and validation sets.

def create_sets(seed: int = 52293):
    """
    Splits the data into training and validation sets, preventing data leakage.
    Returns Ray object IDs for each set.
    """
    data_all = ray.get(data_all_id)
    labels_all = ray.get(labels_all_id)
    # !!! prevent leakage
    np.random.seed(seed)
    trainsize, valsize = int(len(data_all)*0.8), int(len(data_all)*0.2)
    set_sizes = [trainsize, valsize, int(len(data_all)-trainsize-valsize)]
    # randomly assign to train set
    rand_idxs = np.random.choice(
        np.arange(0, len(data_all)), size=set_sizes[0], replace=False)
    data = np.array([data_all[i] for i in rand_idxs])
    labels = np.array([labels_all[i] for i in rand_idxs])
    # put into ray data handler. Torch dataset does not allow large arguments
    data_id = ray.put(data)
    labels_id = ray.put(labels)
    # create val set of remaining
    remaining = list(set(np.arange(0, len(data_all))) - set(rand_idxs))
    rand_idxs_val = np.random.choice(
        remaining, size=set_sizes[1], replace=False)
    data_val = np.array([data_all[i] for i in rand_idxs_val])
    labels_val = np.array([labels_all[i] for i in rand_idxs_val])
    data_val_id = ray.put(data_val)
    labels_val_id = ray.put(labels_val)
    return data_id, data_val_id, None, labels_val_id


#%% CustomDataset for image and sinogram data

rotater = T.RandomRotation(degrees=(0, 180))
affine_transfomer = T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
inverter = T.RandomInvert()

class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset for handling RASER image and sinogram data with optional normalization and padding.
    """
    def __init__(self, image_id, data_id, labels_id, config: RaserConfig, transform: bool = None, target_transform: bool = None):
        self.image = ray.get(image_id)
        self.data = ray.get(data_id)
        self.labels = ray.get(labels_id)
        self.config = config
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, idx: int):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            # Add transform logic if needed
            pass
        if self.target_transform:
            # Add target_transform logic if needed
            pass
        return x, y

#%%Training
def train(config: RaserConfig, checkpoint_dir: str = None, raytune: bool = False):
    """
    Trains the neural network model using the provided configuration.
    Supports Ray Tune for hyperparameter optimization.
    Returns the trained model and error lists.
    """
    EPOCHS = config.epochs
    LR = config.LR
    BATCH_SIZE = config.batch_size

    if 'Unet' in config.arch:
        model = Unet()
    elif 'Autoencoder' in config.arch:
        model = Autoencoder()
    else:
        raise ValueError(f"Unknown architecture: {config.arch}")
    
    model.to(DEVICE)

    if config.loss == 'MSE':
        criterion = nn.MSELoss()
    elif config.loss == 'MAE':
        criterion = nn.L1Loss()
    elif config.loss == 'Huber':
        criterion = nn.HuberLoss()
    else:
        raise ValueError(f"Unknown loss function: {config.loss}")

    if config.optimizer == 'SGD': 
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.LR, momentum=.9, weight_decay=config.WD)
    elif config.optimizer == 'adam': 
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WD)
    elif config.optimizer == 'adamw': 
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WD)
    elif config.optimizer == 'adagrad': 
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config.LR, weight_decay=config.WD)

    if(config.Scheduler):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100)


    #%% Models

    data_id, data_val_id , data_test_id, labels_val_id = create_sets()

    dataset_train = CustomDataset(image_all_id, data_id, data_val_id, config, transform=True)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    dataset_val = CustomDataset(image_all_id, data_test_id, labels_val_id, config)
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    error_train = []
    error_val = []

    for epoch in range(EPOCHS):
        for mode, dataloader in [("train", train_loader), ("val", val_loader)]:
            if mode == "train":
                model.train()
            else:
                model.eval()

            runningLoss = 0
            total = 0

            for i_batch, (inputs, targets, _) in enumerate(dataloader):
                inputs, targets = inputs.float(), targets.float()
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                if 'TPIfCNN'  in config.arch:
                    if(config.TPI_split == config.NR_TPIS ):
                        outputs = model(inputs.unsqueeze(1))
                    else:
                        outputs = model(inputs)

                elif 'TPIfComCNN'  in config.arch:
                    if(config.TPI_split == config.NR_TPIS ):
                        outputs = model(inputs.unsqueeze(1))
                    else:
                        outputs = model(inputs)
                elif 'Unet' in config.arch:
                        outputs = model(inputs.unsqueeze(1))
                elif 'Autoencoder' in config.arch:
                        outputs = model(inputs.unsqueeze(1))
                else: 
              
                    outputs = model(inputs)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, targets)

                runningLoss += loss.item() * inputs.shape[0]
                total += inputs.shape[0]

                if mode == "train":
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                if mode == 'val':
                    if(config.Scheduler):
                        scheduler.step(loss)

            (error_train if mode == "train" else error_val).append(
                runningLoss / total)

        # Remove Ray Tune checkpoint/report if not using Ray Tune
        if raytune:
            # Here we save a checkpoint. It is automatically registered with
            # Ray Tune and will potentially be passed as the `checkpoint_dir`
            # parameter in future iterations.
            try:
                import ray.tune as tune
                from ray.air import session, Checkpoint
            except ImportError:
                raise ImportError("ray.tune is not installed. Please install ray[tune] to use Ray Tune features.")
            checkpoint = Checkpoint.from_dict({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            })
            session.report(
                {"loss": error_val[-1], "loss_train": error_train[-1]},
                checkpoint=checkpoint
            )

        if (epoch) % 1 == 0 and not raytune:
            print('Epoch #{}'.format(epoch))
            print('Train error: ', round(error_train[-1], 6))
            print('Val error: ', round(error_val[-1], 6))
            # print('Time epoch: ', round(end_t - start_t))

    torch.save(model.state_dict(), OUTPATH / 'model.pt')
    import json
    with open(OUTPATH /'config.json', 'w') as f:
        json.dump(config, f)

    return model, error_train, error_val
#%% Model evaluation and testing function
def test_best_model(best_trial = None, raytune: bool = False, model = None, config: RaserConfig = None, error_train = None, error_val = None, eval_only: bool = False):
    """
    Loads the best model (optionally from Ray Tune), evaluates on the test set, and visualizes results.
    Plots loss curves and prediction vs. target images and sinograms.
    """
    if raytune:
        config = best_trial.config
    else:
        config = config

    if raytune or eval_only:
        if 'Unet' in config.arch:
            model = Unet()
         
        model.to(DEVICE)
        if raytune: 
            checkpoint_path = os.path.join(
            best_trial.checkpoint.value, "checkpoint")
            model_state, optimizer_state = torch.load(checkpoint_path)
            model.load_state_dict(model_state)
        if eval_only: 
            model.load_state_dict(torch.load(PATH/"outputs"/  model_String / "model.pt"))  
        
        model.to(DEVICE)
    model.eval()

    if config.loss == 'MSE':
        criterion = nn.MSELoss()
    elif config.loss == 'MAE':
        criterion = nn.L1Loss()
    elif config.loss == 'Huber':
        criterion = nn.HuberLoss()
    else:
        raise ValueError(f"Unknown loss function: {config.loss}")

    
   
    dataset_test = CustomDataset(test_images_id, input_all_id,test_all_id, config, transform = False )
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=config.batch_size, shuffle=False, drop_last=True)
    



    error_test = []
    runningLoss = 0
    total = 0
    for i_batch, (inputs, targets, in_1d) in enumerate(test_loader):
        inputs, targets = inputs.float(), targets.float()
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        if 'TPIfCNN'  in config.arch:
            if(config.TPI_split == config.NR_TPIS ):
                outputs = model(inputs.unsqueeze(1))
            else:
                outputs = model(inputs)
       
        elif 'TPIfComCNN'  in config.arch:
            if(config.TPI_split == config.NR_TPIS ):
                outputs = model(inputs.unsqueeze(1))
            else:
                outputs = model(inputs)
        else: 
          
            outputs = model(inputs.unsqueeze(1))
        outputs = outputs.squeeze(1)
        loss = criterion(outputs, targets)
        runningLoss += loss.item() * inputs.shape[0]
        total += inputs.shape[0]
        error_test.append(runningLoss/total)

    print('error test: ', np.mean(error_test))


    if not raytune:
        plt.figure()
        plt.plot(error_train, label="Train Error")
        plt.plot(error_val, label="Val Error")
        plt.plot(config.epochs-1, np.mean(error_test),
                 'x', label="Test Error")
        plt.xlabel('Epoch #')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training loss')
        if not eval_only: plt.savefig(
            OUTPATH / 'loss_DR_Z2_{}_id{}.png')
        plt.show()
        
    
    
    # Get the data
    idx = 0
    original = targets[idx].cpu().detach().numpy()
    prediciton = outputs.squeeze(1).cpu().detach().numpy()[idx] 
    model_input = inputs[idx].cpu().detach().numpy() 
    
    # added (mob:)
    input_1dmodel = in_1d[idx].cpu().detach().numpy() 
    angles = input_1dmodel[:,201]
    in_Img_1d = iradon(np.transpose(input_1dmodel[:,:200]), theta=angles, output_size = 48,filter_name='cosine')
    
    # Remove padding
    # Define the original shape of the array
    if(config.input == "Sinogram"):
        padded_shape = (80, 32)
        original_shape = (67, 30)
        
        
        # Compute the slice ranges for each dimension
        row_slice = slice(padded_shape[0] - original_shape[0], None)
        col_slice = slice(padded_shape[1] - original_shape[1], None)
        
        # Extract the original array
        original = original[row_slice, col_slice]
        prediciton = prediciton[row_slice, col_slice]
        model_input = model_input[row_slice, col_slice]
    
    # Unpad the images
    elif(config.input == 'Iradon'):
        padded_shape = (48, 48)
        original_shape = (44, 44)

        original = unpad_Image(original,padded_shape,original_shape)
        prediciton = unpad_Image(prediciton,padded_shape,original_shape)
        model_input = unpad_Image(model_input,padded_shape,original_shape)

        
    
    # SSIM of similar structure analysis
    ssim_inp = ssim(original, model_input, data_range=original.max() - original.min())
    ssim_pred = ssim(original, prediciton, data_range=original.max() - original.min())
    
    # The MSE of sinograms
    mse_inp = image_mse(original, model_input)
    mse_pred = image_mse(original, prediciton)


    # Plot one figure
    #extracted_image = prediciton[13:80, 2:32]
    #print(extracted_image.shape)
    #plt.imshow(extracted_image)
    f1 = plt.figure()
    
    ax0 = f1.add_subplot(1, 4, 1)
    ax0.title.set_text('Input 1D')
    plt.imshow(in_Img_1d, cmap = 'gray')
    
  
    ax1 = f1.add_subplot(1, 4, 2)
    ax1.title.set_text('Output 1D \n & Input 2D')
    ax1.set_xlabel(f'SSIM: {ssim_inp:.3f} \n MSE: {mse_inp:.3f}')
    ax1.set_yticks([]) # Delete the y-axis
    ax1.set_yticklabels([]) # Deactivate this next
    plt.imshow(model_input, cmap = 'gray')
    
    ax2 = f1.add_subplot(1, 4, 3)
    ax2.title.set_text('Output 2D')
    ax2.set_xlabel(f'SSIM: {ssim_pred:.3f} \n MSE: {mse_pred:.3f}')
    ax2.set_yticks([]) # Delete the y-axis
    ax2.set_yticklabels([]) # Deactivate this next
    plt.imshow(prediciton, cmap = 'gray')
    # plt.imshow(a, cmap='gray')
    
    
    ax3 = f1.add_subplot(1, 4, 4)
    ax3.title.set_text('Target')
    ax3.set_yticks([]) # Delete the y-axis
    ax3.set_yticklabels([])
    plt.imshow(original, cmap = 'gray')
    
    plt.tight_layout()
    SAVEPATH = PATH / "outputs" / "Moritz" / 'output.svg'
    plt.savefig(SAVEPATH)
    
    
       
    if not eval_only: f1.figure.savefig(
        OUTPATH / 'prediction_id{}_{}.png')
    
    # Reconstruct and compare images from sinograms -----------------------------------------
    # load angels
    
    
    if(config.input == "Sinogram" and config.output == 'Sinogram'):
        
        # Reconstruct the image
        reconstruction_original = iradon(original, theta=angles, output_size = 44,filter_name='cosine')
        reconstruction_prediction = iradon(prediciton, theta=angles, output_size = 44,filter_name='cosine')
        reconstruction_input = iradon(model_input, theta=angles, output_size = 44,filter_name='cosine')
        
        reconstruction_original = (reconstruction_original - reconstruction_original.min()) / (reconstruction_original.max() - reconstruction_original.min())
        reconstruction_prediction = (reconstruction_prediction - reconstruction_prediction.min()) / (reconstruction_prediction.max() - reconstruction_prediction.min())
        reconstruction_input = (reconstruction_input - reconstruction_input.min()) / (reconstruction_input.max() - reconstruction_input.min())
        
        # Get the original image to compare with
        original_image = test_images[-1] # The last one in the set
        original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
    
        # Image metrics
        ssim_img_inp = ssim(original_image, reconstruction_input, data_range=original_image.max() - original_image.min())
        ssim_img_pred = ssim(original_image, reconstruction_prediction, data_range=original_image.max() - original_image.min())
        ssim_img_origin = ssim(original_image, reconstruction_original, data_range=original_image.max() - original_image.min())
        
        #MSE
        mse_img_inp = image_mse(original_image, reconstruction_input)
        mse_img_pred = image_mse(original_image, reconstruction_prediction)
        mse_img_original = image_mse(original_image, reconstruction_original)
        # plot the images
        f2 = plt.figure()
        
      
        ax12 = f2.add_subplot(1, 4,1)
        ax12.title.set_text('Original')
        #ax1.set_xlabel(f'SSIM: {ssim_inp:.2f}')
        plt.imshow(original_image, cmap = 'gray')
        
        ax22 = f2.add_subplot(1, 4,2)
        ax22.title.set_text('Target')
        ax22.set_xlabel(f'SSIM: {ssim_img_origin:.3f} \n MSE: {mse_img_original:.3f}')
        plt.imshow(reconstruction_original, cmap = 'gray')
        # plt.imshow(a, cmap='gray')
        
        
        ax32 = f2.add_subplot(1, 4,3)
        ax32.title.set_text('Input')
        ax32.set_xlabel(f'SSIM: {ssim_img_inp:.3f} \n MSE: {mse_img_pred:.3f}')
        plt.imshow(reconstruction_input, cmap = 'gray')
        
        ax42 = f2.add_subplot(1, 4,4)
        ax42.title.set_text('Prediction')
        ax42.set_xlabel(f'SSIM: {ssim_img_pred:.3f} \n MSE: {mse_img_inp:.3f}')
        plt.imshow(reconstruction_prediction, cmap = 'gray')
        
        
           
        if not eval_only: f2.figure.savefig(
            OUTPATH / 'images_id{}_{}.png')
        
       
    return np.mean(error_test)

#%% Train and test

# eval_only
error = test_best_model(config = initial_config, 
                        error_train = [0], error_val = [0], eval_only=True)
