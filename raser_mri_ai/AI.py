#!/usr/bin/env python3
"""
RASER AI project main training and evaluation script.
This script handles data loading, model training, evaluation, and result visualization for the RASER MRI AI project.
"""
import os
import numpy as np
import torch
import ray
import json
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from skimage.metrics import structural_similarity as ssim
from raser_mri_ai.NN_architectures import TPIfCNN, TPIfComCNN  # Import only the needed classes
from raser_mri_ai.utility_functions import *
from raser_mri_ai.models.config_models import RaserConfig, CNNConfig
from pathlib import Path
from ray import tune

RAYTUNE = False  # Set to True to enable Ray Tune hyperparameter search
DEVICE = 'cuda'  # Device to use for training (cuda or cpu)

# Hyperparameters and configuration dictionary
initial_config = RaserConfig(
    os="Linux",
    id="Test",
    Dataset="100_images",
    set="ExampleSet",
    Testingset="100_images",
    arch="TPIfCNN",
    Testing=0,
    NR_TPIS=16,
    Scheduler=False,
    TPI_included=False,
    TPI_split=16,
    LR=4.1445548340013715e-05,
    WD=1.0129890808702106e-06,
    epochs=1,
    batch_size=60,
    input_shape=201,
    output_shape=67,
    loss="MSE",
    optimizer="adam",
    normalization="layer",
    cnn=CNNConfig(
        outlayer="exprelu",
        activation="relu",
        kernel_size=8,
        dilation=0,
        dropout_conv=0.1,
        dropout_fc=0.2,
        num_layers=5,
        num_layers_conv=3,
        num_layers_lstm=2,
        filters=64,
        hidden=1024,
        stride=1,
        pool_size=2,
        padding=0,
        smoothing=False,
        TPI_included=False,
        smoothfunction="Softplus"
    )
)

# Set up output directory for saving models and results
PATH = Path("./")
model_String = initial_config.os +'_' + initial_config.arch + '_' + str(initial_config.id)
OUTPATH = PATH / "outputs" / model_String
# Check if a output folder exist with the current id, if it doesnt a new folder is created
if not os.path.exists(OUTPATH):
    os.makedirs(OUTPATH)

# Load and preprocess training and testing data
#Training
data_all = np.load(PATH  /"Data/Processedata"/  initial_config.Dataset /"data.npy")
labels_all = np.load(PATH  /"Data/Processedata"/  initial_config.Dataset /'labels.npy')
data_all = data_all.reshape([data_all.shape[0]*data_all.shape[1],1,202])
labels_all = labels_all.reshape(-1, labels_all.shape[-1])

labels_all_id = ray.put(labels_all)
#Testing
input_all = np.load(PATH  /"Data/Processedata"/  initial_config.Testingset /'data.npy')
test_all = np.load(PATH  /"Data/Processedata"/  initial_config.Testingset /'labels.npy')
input_all = input_all.reshape([input_all.shape[0]*input_all.shape[1],1,202])
test_all = test_all.reshape(-1, test_all.shape[-1])
test_images = np.load(PATH  /"Data/Processedata"/  initial_config.Testingset /'images.npy')

# send to ray

data_all_id = ray.put(data_all)
input_all_id = ray.put(input_all)
test_all_id = ray.put(test_all)

def create_sets(data_all_id, labels_all_id, seed: int = 52293):
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

    return data_id, labels_id, data_val_id, labels_val_id, #data_test_id, labels_test_id


# %%
class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset for handling RASER data with optional normalization and shuffling.
    """
    def __init__(self, data_id, labels_id, config: RaserConfig, transform: bool = None, target_transform: bool = None):
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


#%% Model classes
  


# %%Train

def train(config: RaserConfig, checkpoint_dir: str = None, raytune: bool = False):
    """
    Trains the neural network model using the provided configuration.
    Supports Ray Tune for hyperparameter optimization.
    Returns the trained model and error lists.
    """

    EPOCHS = config.epochs
    LR = config.LR
    BATCH_SIZE = config.batch_size
    BATCH_SIZE = config.batch_size

    if 'TPIfCNN' in config.arch:
        model = TPIfCNN(input_shape=(int(round(config.NR_TPIS/config.TPI_split + .4)), config.input_shape), meta_shape=1, out_shape=config.output_shape, config=config)
    elif 'TPIfComCNN' in config.arch:
        model = TPIfComCNN(input_shape=(int(round(config.NR_TPIS/config.TPI_split + .4)), config.input_shape), meta_shape=1, out_shape=config.output_shape, config=config)
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

    if config.Scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100)

    data_id, labels_id, data_val_id, labels_val_id = create_sets(
        data_all_id, labels_all_id)

    dataset_train = CustomDataset(data_id, labels_id, config, transform=True)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    dataset_val = CustomDataset(data_val_id, labels_val_id, config)
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

            for i_batch, (inputs, targets) in enumerate(dataloader):
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
                  
                    outputs = model(inputs)
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

        if raytune:
            # Here we save a checkpoint. It is automatically registered with
            # Ray Tune and will potentially be passed as the `checkpoint_dir`
            # parameter in future iterations.
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)
            tune.report(loss=error_val[-1], loss_train=error_train[-1])

        if (epoch) % 1 == 0 and not raytune:
            print('Epoch #{}'.format(epoch))
            print('Train error: ', round(error_train[-1], 6))
            print('Val error: ', round(error_val[-1], 6))
            # print('Time epoch: ', round(end_t - start_t))

    torch.save(model.state_dict(), OUTPATH / 'model.pt')
    import json
    with open(OUTPATH  /'config.json', 'w') as f:
        json.dump(config, f)

    return model, error_train, error_val

# %% Testing


def test_best_model(best_trial = None, raytune: bool = False, model = None, config: RaserConfig = None, error_train = None, error_val = None) -> float:
    """
    Loads the best model (optionally from Ray Tune), evaluates on the test set, and visualizes results.
    Plots loss curves and prediction vs. target profiles.
    Returns the mean test error.
    """
    if raytune:
        config = best_trial.config
    else:
        config = config

    if raytune:
        
        if 'TPIfCNN' in config.arch:
            model = TPIfCNN(input_shape=(int(config.NR_TPIS/config.TPI_split), config.input_shape), meta_shape=1, out_shape=config.output_shape, config=config)
        if 'TPIfComCNN' in config.arch:
            model = TPIfComCNN(input_shape=(int(round(config.NR_TPIS/config.TPI_split + .4)), config.input_shape), meta_shape=1, out_shape=config.output_shape, config=config)
         
        model.to(DEVICE)
        checkpoint_path = os.path.join(
            best_trial.checkpoint.value, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint_path)
        model.load_state_dict(model_state)
        model.to(DEVICE)
    model.eval()

    if config.loss == 'MSE': criterion = nn.MSELoss()
    if config.loss == 'MAE': criterion = nn.L1Loss()
    if config.loss == 'Huber': criterion = nn.HuberLoss()


    dataset_test = CustomDataset(input_all_id, test_all_id, config, transform=False)
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=config.batch_size, shuffle=False, drop_last=True)
    


   
    error_test = []
    runningLoss = 0
    total = 0
    for i_batch, (inputs, targets) in enumerate(test_loader):
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
          
            outputs = model(inputs)

        loss = criterion(outputs, targets)
        runningLoss += loss.item() * inputs.shape[0]
        total += inputs.shape[0]

        error_test.append(runningLoss/total)
    # error_test = criterion(out, torch.Tensor(test_labels).to(DEVICE)).item()
    # print("Test Accuracy:", sum(out_classes == test_labels) / len(test_labels))
    print('error test: ', np.mean(error_test)) # Print test error
    print('NR of TPI: '  + str(config.NR_TPIS/config.TPI_split)) # Print the number of spectra used

    if not raytune: # If not raytune, save the model and plot the loss
        plt.figure()
        plt.plot(error_train, label="Train Error")
        plt.plot(error_val, label="Val Error")
        plt.plot(config.epochs-1, np.mean(error_test),
                 'x', label="Test Error")
        plt.xlabel('Epoch #')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training loss')
        plt.savefig(
            OUTPATH / 'loss_DR_Z2_{}_id{}.png')
        plt.show()


    # Plot one figure
    idx = 0
    fig1 = plt.axes()

    initial_config

    fig1.plot(outputs.cpu().detach().numpy()[idx], label="Predicted Profile")
    fig1.plot(targets.cpu().detach().numpy()[idx], label="Target Profile")
    fig1.legend()

    fig1.set_title('Model Profile and Target Profile')
    # TODO information about dataset
   
    fig1.figure.savefig(
        OUTPATH / 'prediction_id{}_{}.png')
    
# Plot several figures
    if(config.TPI_split == config.NR_TPIS):
        print(inputs.cpu().detach().numpy().shape)
        input_plotData = inputs.cpu().detach().numpy().reshape((config.batch_size, 1, 201))
    else:
        print(inputs.cpu().detach().numpy().shape)
        input_plotData = inputs.cpu().detach().numpy()
        
    target_plotData = targets.cpu().detach().numpy()
    output_plotData = outputs.cpu().detach().numpy()
        
    ## See some images
    number_Imgs = int(config.batch_size/30)
    input_imageData = np.reshape(input_plotData, [number_Imgs,30,201])
    angles_imgData = input_imageData[:,:,-1] # get the angles
    input_imageData = input_imageData[:,:,:200] # remove the angles from the data
    
    output_imageData = np.reshape(output_plotData, [number_Imgs,30,67])
    target_imageData = np.reshape(target_plotData, [number_Imgs,30,67])
    

    idx = 0 # Random index
    output_sinogram = np.transpose(output_imageData[idx]) # Transpose to get the right shape
    target_sinogram = np.transpose(target_imageData[idx])
    input_sinogram = np.transpose(input_imageData[idx])
    angles = angles_imgData[idx]
    
    reconstruction_sart = iradon(output_sinogram, theta=angles, output_size = 44,filter_name='cosine') # Reconstruct the image
    reconstruction_targ = iradon(target_sinogram, theta=angles, output_size = 44,filter_name='cosine')
    reconstruction_inp = iradon(input_sinogram, theta=angles,filter_name='cosine')
    
    original_image = test_images[idx]
    
    # Get the SSIM
    ssim_img = ssim(original_image, reconstruction_sart, data_range=original_image.max() - original_image.min())
    ssim_sino = ssim(target_sinogram, output_sinogram, data_range=target_sinogram.max() - target_sinogram.min())
    
    f, axarr = plt.subplots(2,3)
    
    # Plot the images
    axarr[0,0].imshow(reconstruction_inp, cmap='gray')
    axarr[0,1].imshow(reconstruction_sart, cmap='gray')
    axarr[0,1].set_xlabel(f'SSIM: {ssim_img:.2f}')
    axarr[0,2].imshow(original_image, cmap='gray')
    
    axarr[1,0].imshow(input_sinogram, cmap='gray')
    axarr[1,1].imshow(output_sinogram, cmap='gray')
    axarr[1,1].set_xlabel(f'SSIM: {ssim_sino:.2f}')
    axarr[1,2].imshow(target_sinogram, cmap='gray')
    
    axarr[0,0].title.set_text('RASER Image')
    axarr[0,1].title.set_text('Predicted Image')
    axarr[0,2].title.set_text('Target Image')
    
    axarr[1,0].title.set_text('RASER Sinogram')
    axarr[1,1].title.set_text('Predicted Sinogram')
    axarr[1,2].title.set_text('Target Sinogram')
    
    f.tight_layout()
    
    f.savefig( # Save the figure
         OUTPATH / 'Input_output_target_id{}_{}.png')

    return np.mean(error_test)

if(RAYTUNE == False):
        
   model, error_train, error_val = train(config=initial_config)
   error = test_best_model(model=model, config = initial_config, error_train = error_train, error_val = error_val)

# %% Function for plotting results
# Input is the index and ID of the output file which is used to get the model and config file
# The Spectra at the index is put into the model and the result is then ploted together with the
# reference spectra
# To-do add functionality for more nn archytypes
# To do add input set so it works for testing aswell
def plotResults(idx: int, ID, input_data, targets) -> None:
    """
    Plots the predicted and target spectra for a given input index and model ID.
    """

    input = input_data[idx]
    config_file = open(PATH + '/outputs/' + ID + '/config.json', "r")
    config = json.loads(config_file.read())
    input_tensor = torch.from_numpy(input)
    input_tensor = input_tensor.float()
    input_tensor = input_tensor.unsqueeze(0)
    target = torch.from_numpy(targets[idx])
    data_val_id = ray.put(input_tensor)
    labels_val_id = ray.put(target)
    
    new_data = CustomDataset(data_val_id, labels_val_id, config)
    single_input = new_data[0][0]
    single_input = torch.from_numpy(single_input)
    single_input = single_input.float()
    single_input = single_input.to(DEVICE)
    single_input = single_input.unsqueeze(0)
    model_path = (PATH + '/outputs/' + str(ID) + '/model.pt')
    if 'TPIfCNN' in config['arch']:
        model = TPIfCNN(input_shape=(int(config["NR_TPIS"]/config['TPI_split']),config['input_shape']), meta_shape=1, out_shape=config['output_shape'], config=config)

        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.to(DEVICE)
        outputs = model(single_input)
    elif 'TPIfComCNN' in config['arch']:
        model = TPIfComCNN(input_shape=(int(config["NR_TPIS"]/config['TPI_split']),config['input_shape']), meta_shape=1, out_shape=config['output_shape'], config=config)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.to(DEVICE)
        outputs = model(single_input)
    else:
        raise ValueError(f"Unknown architecture: {config['arch']}")

    plt.figure()
    initial_config

    plt.plot(outputs.cpu().detach().numpy(), label="Predicted Profile")
    plt.plot(targets[idx], label="Target Profile")

    if(config['TPI_included']):
        plt.title('Index:' + str(idx) + ' Arch:' + config['arch'] + ' TPI: ' + str((3e+16) ))
    else: plt.title('Index:' + str(idx) + ' Arch:' + config['arch']  )

    plt.legend()


# %% HyperOpt with ray tune

# https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-taking-union-of-dictiona


def merge_two_dicts(x: dict, y: dict) -> dict:
    """
    Merges two dictionaries and returns the result.
    """
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z


# Function to train and evaluate models with different TPI splits
def trainMultiple() -> None:
    """
    Trains several models with different TPI splits and prints their errors.
    Saves each model in the outputs folder.
    """
    config = initial_config.copy()
    tpi_Splits = [16,8,4,2,1]
    train_Errors = np.empty(len(tpi_Splits))
    val_Errors = np.empty(len(tpi_Splits))
    test_Errors = np.empty(len(tpi_Splits))
    for i, d in enumerate(tpi_Splits):
       
        config['TPI_split'] = d
        config['id'] = str(int(config['id'])+ 1)
        OUTPATH = PATH + '/outputs/' + config['os'] +'_' + config['arch'] + '_' + str(config['id'])

        # Check if a output folder exist with the current id, if it doesnt a new folder is created
        if not os.path.exists(OUTPATH):
            os.makedirs(OUTPATH)
        model, train_List, val_List = train(config)
        train_Errors[i] = train_List[config['epochs']-1] 
        val_Errors[i] = val_List[config['epochs']-1] 
        test_Errors[i] = test_best_model(model=model,config=config, error_train=train_List, error_val=val_List)
   
    print(train_Errors)
    print(val_Errors)
    print(test_Errors)

def main():
    if RAYTUNE == False:
        # Typical training and evaluation run
        model, error_train, error_val = train(config=initial_config)
        error = test_best_model(model=model, config=initial_config, error_train=error_train, error_val=error_val)

if __name__ == "__main__":
    main()


