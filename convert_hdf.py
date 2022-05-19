# External libraries
import pathlib
import wfdb
import numpy as np
import re
from tqdm import tqdm
import os
import pandas as pd
import h5py

# Internal libraries

# Constants
from constants import PTBXL_HZ

# Function
def convert(args):
    """ Converts wfdb files into hdf5 files to be used in Lima model. """
    
    # Check if file is present
    folder = pathlib.Path(args.save_hdf).with_suffix('')
    if os.path.exists(args.save_hdf) or os.path.exists(folder):
        if args.replace:
            tqdm.write("Note: Replacing previously stored data.")
        else:
            raise Exception('Data already exist. Change save location or set "--replace" to True.')

    # Create folder if not existing
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Create list of data files
    if args.recursive:
        data_files = list(pathlib.Path(args.data).glob('**/*.hea'))
    else:
        data_files = list(pathlib.Path(args.data).glob('*.hea'))

    # Check that files are present
    if len(data_files) == 0:
        raise FileNotFoundError("No files found.")

    # Open CSV
    metadata = pd.read_csv(args.metadata)

    # Remove nan's from age
    metadata.dropna(subset = [args.age_col], inplace=True)

    # Set path stem in metadata
    pattern = re.compile('(?<=\/)(.*?)(?=\/)')
    metadata['stem'] = metadata.filename_hr.str.extract(pattern)
    metadata['stem_n'] = metadata.groupby('stem')['stem'].transform('size')
    stems = metadata["stem"].unique()

    # Check if there should be a outer progress bar
    if not args.folder:
        stems = range(1)
    else:
        stems = tqdm(stems)

    # Go over each stem
    for stem in stems:
        # Create list for np arrays
        signal_data = []
        exam_data = []
        age_data = []
        # _data = []

        # Check if data is to be extracted into folders or single file
        if not args.folder:
            files = data_files
            stem = "data"
        else:
            # Find data files with given stem
            files = [x for x in data_files if x.match('*/' + str(stem) + '/*')]
        
        # Check if file is present
        if len(files) == 0:
            continue
        
        # Convert files
        for file in tqdm(files):

            # Open record
            record = wfdb.rdrecord(file.__str__()[:-4])

            # Extract data
            signal = extract_signal(record)
            exam_id = record.record_name
            exam_id = int(re.sub(PTBXL_HZ, '', exam_id))

            # Check if data is missing (from metadata)
            if exam_id not in metadata[args.id_col].values:
                continue

            # Extract data
            df = metadata.query(f"{args.id_col} == {exam_id}")

            # Append data
            signal_data.append(signal)
            exam_data.append(exam_id)
            age_data.append(int(df[args.age_col].values[0]))

        # Combine stack
        signal_data = np.stack(signal_data, axis=0)

        exam_data = np.stack(exam_data, axis=0)

        age_data = np.stack(age_data, axis=0)

        # norm_data = np.stack(norm_data, axis=0)

        # Save data
        save_hdf(signal_data, exam_data, age_data, stem, folder, args.save_hdf)

    # Save 
    save_csv(metadata, args.save_csv, args.id_col)

    # Status
    tqdm.write("\nData converted!\n")

# Subfunctions
def extract_signal(record):
    """ Extracts and trims wfdb signal from a given record """
    # Open record
    signal = np.array(record.p_signal)

    # Cleave signal into correct size (4096) - Trim on both sides
    signal_cut = signal.shape[0] - 4096
    signal_top = int(signal.shape[0] - (signal_cut/2))
    signal_bottom = int(signal_cut/2)
    signal = signal[signal_bottom:signal_top]

    # Return
    return(signal)

def save_hdf(tracings, exams, age, stem, folder, hdf_save):
    """ Creates a h5py file strcuture. """

    # Extract first dimension
    n_items = tracings.shape[0]

    # Define save path 
    save_path = str(folder) + "/" + str(stem) + ".hdf5"

    # Save hdf5 
    with h5py.File(save_path, 'w') as f:
        f.create_dataset("tracings", (n_items, 4096, 12), data=tracings, dtype='<f4')
        f.create_dataset("exam_id", data=exams, dtype='int64')
        f.create_dataset("true_age", data=age, dtype='int64')
        # f.create_dataset("norm", data=norm, dtype='bool')


def save_csv(metadata, csv_save, index):
    """ Save CSV. """
    
    # Set index
    metadata = metadata.set_index(index)
    # Save
    metadata.to_csv(csv_save)