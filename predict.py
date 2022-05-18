# External libraries
import os
import torch
import h5py
import json
import numpy as np
from tqdm import tqdm
import pandas as pd

# Internal libraries
from resnet import ResNet1d

# Constants
from constants import N_LEADS

# Function
def predict(args):
    """ Predict age using predefined model. """
    
    # Check if predictions already exists
    if os.path.exists(args.save):
        if args.replace:
            tqdm.write("Note: Replacing previously stored predictions.")
        else:
            raise Exception('Predictions already exist. Change save location or set "--replace" to True.')
    
    # Set random seed
    torch.manual_seed(args.seed)

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Define model parameters
    model_path = os.path.join(args.model, 'model.pth')
    checkpoints = torch.load(model_path, map_location=lambda storage, loc: storage)
    config_path = os.path.join(args.model, 'config.json')
    
    # Load config info
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load model
    model = ResNet1d(input_dim=(N_LEADS, config['seq_length']),
                    blocks_dim=list(zip(config['net_filter_size'], config['net_seq_length'])),
                    n_classes=1,
                    kernel_size=config['kernel_size'],
                    dropout_rate=config['dropout_rate'])
    model.load_state_dict(checkpoints["model"])
    model = model.to(device)

    # Load data
    data = h5py.File(args.data, 'r')
    traces = data[args.traces_dset]
    ids = data[args.ids_dset]
    age = data[args.age_dset]
    n_total = len(traces)

    # Prepare data for mo model
    model.eval() # Turn on prediction mode
    n_total, n_samples, n_leads = traces.shape
    n_batches = int(np.ceil(n_total/args.batch_size))

    # Predict 
    predicted_age = np.zeros((n_total,)) # empty ds
    end = 0 # Setting
    for i in tqdm(range(n_batches)): # Loop
        
        # Set start (and calculate new /end)
        start = end
        end = min((i + 1) * args.batch_size, n_total) # Min skips last one

        # Compute gradient
        with torch.no_grad():
            # Create tensor from data
            x = torch.tensor(traces[start:end, :, :]).transpose(-1, -2)

            # Send to GPU/CPU
            x = x.to(device, dtype=torch.float32)
            
            # Make prediciton (feed into model)
            y_pred = model(x)
        
        # Add to final data
        predicted_age[start:end] = y_pred.detach().cpu().numpy().flatten()
        
    # Build dataframe
    df = pd.DataFrame({'ids': ids, 'predicted_age': predicted_age, 'true_age': age} )
    df = df.set_index('ids').sort_index()
    
    # Save
    df.to_csv(args.save)
    tqdm.write("\nPredictions made!\n")

# Subfunctions
