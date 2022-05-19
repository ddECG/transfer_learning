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


def fix_id(args):
    """ Adds ID column to hdf_file. """

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

    # Check that files are present
    if len(args.data) == 0:
        raise FileNotFoundError("No files found.")

    # Open CSV
    metadata = pd.read_csv(args.metadata)

    # Extract tracings from HDF5
    with h5py.File(args.data, "r") as f:
        tracings = np.array(f['tracings'])

    # Create ID data
    id = list(range(1, len(tracings) + 1))
    id = np.array(id)

    # Age column
    age = metadata.age
    age = np.array(age)

    # Combine data in new hdf5 file
    with h5py.File(args.save_hdf, 'w') as f:
        f.create_dataset("tracings", (len(tracings), 4096, 12), data=tracings, dtype='<f4')
        f.create_dataset("exam_id", data=id, dtype='int64')
        f.create_dataset("true_age", data=age, dtype='int64')