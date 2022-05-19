# External libraries
import os
from tqdm import tqdm
import json
import pandas as pd
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Internal libraries
from resnet import ResNet1d
from dataloader import BatchDataloader

# Constants
from constants import N_LEADS, N_CLASSES

# Function
def train(args):
    """ Trains model. """
    # Check if model already exists
    if os.path.exists(args.model):
        if args.replace:
            tqdm.write("Note: Replacing previously stored model.")
        else:
            raise Exception('Model already exist. Change save location or set "--replace" to True.')

    # Create folder if not existing
    if not os.path.exists(args.model):
        os.makedirs(args.model)
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create config
    argument_path = os.path.join(args.model, 'config.json')
    with open(argument_path, "w") as f:
        json.dump(vars(args), f, indent='\t')
    
    # Load checkpoints
    
    # Part 1 - Build data loaders
    tqdm.write("\nPart 1: Building data loaders...")

    # Open data
    metadata, exam_id, ages, traces = open_data(args)

    # Set validation/training mask
    validation_mask, training_mask = data_masking(args, len(metadata))
    
    # Compute weights
    weights = compute_weights(args, ages)

    # Dataloaders
    train_loader = BatchDataloader(traces, ages, weights, bs=args.batch_size, mask=training_mask)
    valid_loader = BatchDataloader(traces, ages, weights, bs=args.batch_size, mask=validation_mask)
    
    tqdm.write("\tData is loaded!\n")

    # Part 2 - Define model
    tqdm.write("Part 2: Defining model...")

    # Define resnet + settings
    model = ResNet1d(input_dim=(N_LEADS, args.seq_length),
                    blocks_dim=list(zip(args.net_filter_size, args.net_seq_length)),
                    n_classes=N_CLASSES,
                    kernel_size=args.kernel_size,
                    dropout_rate=args.dropout_rate)
    
    # Checking training type
    if args.tune:

        # Fine tuning model
        tqdm.write("\tFine tuning existing model.")
        model_path = os.path.join(args.tune_model, 'model.pth')
        checkpoints = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoints["model"])
        model.to(device=device)
    elif args.modify:
        # Modifying
        tqdm.write("Modifying model")
        # Freeze layers
        for param in model.parameters():
            param.requires_grad = False

        # Modify last layer
        model.lin = nn.Linear(5120,1,bias=True)
        model.to(device=device)

        #model.fc = nn.Linear(1000, 8)
        # number_features = model.lin[6].in_features

    else:
        # Creating new model
        tqdm.write("\tCreating new model.")
        model.to(device=device)
    
    tqdm.write("\tModel is defined!\n")

    # Part 3 - Define optimizer
    tqdm.write("Part 3: Defining optimizer...")
    optimizer = optim.Adam(model.parameters(), args.lr)
    tqdm.write("\tOptimizer is defined!\n")

    # Part 4 - Define scheduler
    tqdm.write("Part 4: Defining scheduler...")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience,
                                                    min_lr=args.lr_factor * args.min_lr,
                                                    factor=args.lr_factor)    
    tqdm.write("\tScheduler is defined!\n")

    # Part 5 - Train data
    tqdm.write("Part 5: Training model...\n")
    
    # Prepare training
    start_epoch = 0
    best_loss = np.Inf

    # Create dataframe for training history
    history = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'lr',
                                    'train_accuracy', 'train_error', 
                                    'valid_accuracy', 'valid_error'])
    
    # Epoch (Training) loop
    for epoch in range(start_epoch, args.epochs):

        # Train (and store loss)
        train_loss, train_accuracy, train_error = training(epoch, args.epochs, train_loader, model, device, optimizer)
        valid_loss, valid_accuracy, valid_error = evaluate(epoch, args.epochs, valid_loader, model, device, optimizer)

        # Save model (if better)
        if valid_loss < best_loss:
            torch.save({'epoch': epoch,
                            'model': model.state_dict(),
                            'valid_loss': valid_loss,
                            'optimizer': optimizer.state_dict()},
                        os.path.join(args.model, 'model.pth'))

            # Set new loss
            best_loss = valid_loss

        # Get learning rate
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        
        # Interrupt for minimum learning rate
        if learning_rate < args.min_lr:
            break

        # Training set status
        tqdm.write(f"Training set")
        tqdm.write(f"\tEpoch: {epoch}\n\tTrain Loss {train_loss}\n\t Average accuracy: {train_accuracy}\n\t Average error {train_error}\n")

        # Validation set status
        tqdm.write(f"Validation set")
        tqdm.write(f"\tValid loss: {valid_loss}\n\tLearning rate {learning_rate}\n\t Average accuracy: {valid_accuracy}\n\t Average error {valid_error}")

        # Update history
        history = history.append({"epoch": epoch, "train_loss": train_loss,
                                "valid_loss": valid_loss, "lr": learning_rate,
                                "train_accuracy": train_accuracy, "train_error": train_error,
                                "valid_accuracy": valid_accuracy, "valid_error": valid_error}, ignore_index=True)
        history.to_csv(os.path.join(args.model, 'history.csv'), index=False)        

        # Update learning rate
        scheduler.step(valid_loss)

    tqdm.write("Model trained!")

# Subfunctions
def open_data(args):
    """ Reads metadata and trace data. """

    # Read metadata
    metadata = pd.read_csv(args.metadata, index_col=args.id_col)

    # Read data (HDF5)
    hdf_data = h5py.File(args.data, 'r')
    traces = hdf_data[args.traces_dset]
    exam_id = hdf_data[args.ids_dset]
    # norm = hdf_data["norm"]

    # Reindex data based on exam ids from hdf5 (remove values not present)
    metadata = metadata.reindex(exam_id, fill_value=False, copy=True)

    # Extract age (Important to do this after reindex)
    ages = metadata[args.age_col]

    # Return
    return(metadata, exam_id, ages, traces)

def data_masking(args, n):
    """ Sets validation and training data mask. """

    # Check validation percentage
    if args.validation_percentage < 0.01 or args.validation_percentage > 0.99:
        raise ValueError(f"Validation percentage is {args.validation_percentage * 100}. It must be between 0.01% and 100%.")

    # Calculate number of samples
    n_validation = n * args.validation_percentage
    n_validation = int(round(n_validation, 0))
    
    # Set validation mask
    validation_mask = np.arange(n) <= n_validation

    # Set training mask
    training_mask = ~validation_mask
    ## NOTE: The n_validation is 0-based, and thererefore will cause the following: n=0 == 1 case etc.

    # Return
    return(validation_mask, training_mask)

def compute_weights(args, age):
    """ Compute weights """

    # Calculate unique values (unique), the indecies of the unique (inverse), and the counts (counts)
    unique, inverse, counts = np.unique(age, return_inverse=True, return_counts=True)

    # Set weights as percentage of counts of various ages
    weights = 1 / counts[inverse]
    normalized_weights = weights / sum(weights) # normalize weights
    normalized_weights = len(age) * normalized_weights

    # Return
    return(normalized_weights)

def compute_loss(ages, predicted_ages, weights):
    """ Computes loss. """

    # Calculate difference between real age and predicted ages
    difference = ages.flatten() - predicted_ages.flatten()

    # Calculate loss 
    loss = torch.sum(weights.flatten() * difference * difference)

    # Calculate relative error and accuracy
    relative_error = abs((difference / ages.flatten()) * 100)
    accuracy = 100 - relative_error
    relative_error = torch.sum(relative_error) / len(ages)
    accuracy = torch.sum(accuracy) / len(ages)

    # Return
    return(loss, relative_error, accuracy)    

def training(epoch, n_epochs, train_data, model, device, optimizer):
    """ Trains model on test data. """

    # Put model into training mode
    model.train()

    # Set defaults
    total_loss = 0
    total_accuracy = 0
    total_error = 0
    total_entries = 0

    # Build progress bar
    train_desc = "Epoch: {:2d}/{} | Loss: {:.6f} | Accuracy: {:.1f}% | Error: {:.1f}%"
    train_bar = tqdm(initial=0, leave=True, total=len(train_data),
                    desc=train_desc.format(epoch, n_epochs, 0, 0, 0, 0), position=0)
    
    count = 1
    # Run model
    for traces, ages, weights in train_data:

        # Transpose traces
        traces = traces.transpose(1, 2)

        # Send data to device (CUDA or CPU)
        traces, ages, weights = traces.to(device), ages.to(device), weights.to(device)

        # Set gradients
        model.zero_grad()

        # Predict ages (forward pass)
        predicted_ages = model(traces)

        # Calculate loss
        loss, error, accuracy = compute_loss(ages, predicted_ages, weights)

        # Backwards pass
        loss.backward()

        # Optimize
        optimizer.step()

        # Update stats
        total_loss += loss.detach().cpu().numpy()
        total_accuracy += accuracy.detach().cpu().numpy()
        average_accuracy = total_accuracy / count
        total_error += error.detach().cpu().numpy()
        average_error = total_error / count
        total_entries += len(traces)
        count += 1

        # Update progress bar
        train_bar.desc = train_desc.format(epoch, n_epochs, total_loss / total_entries, accuracy, error)
        train_bar.update(1)
    
    # Close train bar
    train_bar.close()
    return(total_loss / total_entries, average_accuracy, average_error)



def evaluate(epoch, n_epochs, validation_data, model, device, optimizer):
    """ Evaluates model on validation data. """

    # Put model into evaluation mode
    model.eval()

    # Set defaults
    total_loss = 0
    total_entries = 0
    total_accuracy = 0
    total_error = 0

    # Build progress bar
    eval_desc = "Epoch: {:2d}/{} | Loss: {:.6f} | Accuracy: {:.1f}% | Error: {:.1f}%"
    eval_bar = tqdm(initial=0, leave=True, total=len(validation_data),
                    desc=eval_desc.format(epoch, n_epochs, 0, 0, 0, 0), position=0)
    
    count = 1
    # Run model
    for traces, ages, weights in validation_data:

        # Transpose traces
        traces = traces.transpose(1, 2)

        # Send data to device (CUDA or CPU)
        traces, ages, weights = traces.to(device), ages.to(device), weights.to(device)
    
        # Compute gradient  
        with torch.no_grad():

            # Predict (Forward pass)
            predicted_ages = model(traces)

            # Compute loss
            loss, error, accuracy = compute_loss(ages, predicted_ages, weights)

            # Update stats
            total_loss += loss.detach().cpu().numpy()
            total_accuracy += accuracy.detach().cpu().numpy()
            average_accuracy = total_accuracy / count
            total_error += error.detach().cpu().numpy()
            average_error = total_error / count
            total_entries += len(traces)
            count += 1

            # Update progress bar result
            eval_bar.desc = eval_desc.format(epoch, n_epochs, total_loss / total_entries, accuracy, error)
            eval_bar.update(1)
    eval_bar.close()
    
    return(total_loss / total_entries, average_accuracy, average_error)
