# Libraries
import re
import numpy as np
import wfdb
import os
import re
import pathlib
import h5py
import sys

# Convert wfdb files to h5py files
def extract_signal(record):
    
    # Open record
    signal = np.array(record.p_signal)

    # Cleave signal into correct size (4096) - Trim on both sides
    signal_cut = signal.shape[0] - 4096
    signal_top = int(signal.shape[0] - (signal_cut/2))
    signal_bottom = int(signal_cut/2)
    signal = signal[signal_bottom:signal_top]

    # Return
    return(signal)

# Combine h5py files
def combine_h5py():
    pass

# Create h5py file
def create_h5py(tracings, exams, save_location):

    # Extract first dimension
    n_items = tracings.shape[0]

    if os.path.exists(save_location):
        print("Path already exists")
        return 0
    else:
        # Define filename
        with h5py.File(save_location, 'w') as f:
            f.create_dataset("tracings", (n_items, 4096, 12), data=tracings, dtype='<f4')
            f.create_dataset("exam_id", data=exams, dtype='int64')



# Main
def main(dir: str, save_location: str):

    # Constants
    regex_extract_id = '_hr'

    # Create list of files in folder
    files = list(pathlib.Path(dir).glob('*.hea'))

    # Create list for np arrays
    signal_data = []
    exam_data = []

    # Convert files
    for file in files:
        # Open record
        record = wfdb.rdrecord(file.__str__()[:-4])

        # Signal
        signal = extract_signal(record)
        signal_data.append(signal)

        # Exam
        exam_id = record.record_name
        exam_id = int(re.sub(regex_extract_id, '', exam_id))
        exam_data.append(exam_id)

    # Combine stack
    signal_data = np.stack(signal_data, axis=0)
    exam_data = np.stack(exam_data, axis=0)

    # Create h5py file
    create_h5py(signal_data, exam_data, save_location)


#  
if __name__ == "__main__":

    # Extract args
    dir = sys.argv[1]
    save_location = sys.argv[2]
    
    # Run main
    main(dir, save_location)