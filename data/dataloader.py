from itertools import count
import torch
from torch.utils.data import Dataset
import h5py
from pathlib import Path
import numpy as np

class H5Dataset(Dataset):
    """ HDF5 dataset loader.

    Args:
        file_path (str): Path to folder containing several or a single hdf5 file.
        data_cache_size (int): TODO
        transform (optional): TODO - Optional transform to be applied to samples.
    """
    
    # Constructor
    def __init__(self, file_path: str):

        # Variables
        super().__init__()
        self.annotations = []
        self.file_path = file_path
        self.files = None
        self.data_cache = None
        self.len = None

        # Load files
        self.set_files(self.file_path)

        # Set file annotations
        try:
            for file in self.files:
                self.set_annotations(file)
        except TypeError:
            self.set_annotations(self.files)

        # Get length
        self.set_length()

        # Debug
        # print(self.annotations)
        # print(self.get_annotations('exam_id', 1))
        # print(self.annotations[2])
        # print(self.get_annotations(1)[0]['cache'])
        # print(self.get_data(1))
        # print(self.get_data('exam_id', 1))
        # print(self.data_cache)
        
    # Magic methods
    def __getitem__(self, idx: int):
        """
        Gets items and returns as tensor
        """
        # Read data
        data = self.get_data(idx=idx)

        # Return
        return data

    def __len__(self):
        """ Length of dataset """
        return self.len
    
    # Getters
    def get_annotations(self, idx:int):
        """ Extract dict of specific type and at index""" 
        
        # Extract ID
        id = self.annotations[idx]['id']
        
        # Get dict
        dat = [di for di in self.annotations if di['id'] == id]

        # Return
        return(dat)
    
    def get_data(self, idx: int):
        """ Load data into cache """
        # Get file info
        annotation_trace = self.get_annotations(idx)[0]
        annotation_label = self.get_annotations(idx)[1]
        path = annotation_trace['file_path']

        # Check if data in cache
        if annotation_trace['cache'] or annotation_label['cache']:
            dat = self.data_cache
            print("Data is already loaded (Cache is 1)")
            return(dat)
        
        # Extract data (and convert to tensor)
        with h5py.File(path) as f:
            tracings = f['tracings'][idx,:,:]
            tracings = torch.from_numpy(tracings)
            label = f['exam_id'][idx]
            label = torch.from_numpy(np.array(label))
        dat = tracings, label

        # Update cache
        self.set_cache(dat, idx)
        
        # Return
        return(dat)
    
    # Setters
    def set_files(self, file_path):
        """ Finds single file or all hdf5 files within folder. Is not recursive (YET) """
        path = Path(file_path)
        if path.is_dir():
            files = sorted(path.glob('*.hdf5'))
        else:
            files = path

        self.files = files

    def set_cache(self, dat, idx):
        """ Stores data into cache data. Can currently only hold 1 data point """
        # Set cache
        self.data_cache = dat
        
        # Update data annotations
        self.annotations[idx].update({'cache': 1})
        self.annotations[idx + 1].update({'cache': 1}) 

    def set_length(self):
        """ Sets length of dataset """
        self.len = int(len(self.annotations) / 2)
    
    def set_annotations(self, file_path):
        """ Sets data annotations """
        # Set info (remove 1st dimension of traces as this is scan ID)
        with h5py.File(file_path) as file:
            tracings = file['tracings']
            exam_id = file['exam_id']

            # print(tracings.shape)
            for i in range(len(tracings)):
                self.annotations.append({'id': exam_id[i], 'file_path': file_path, 'type': 'tracings', 'shape': tracings[i,:,:].shape, 'cache': 0})
                self.annotations.append({'id': exam_id[i], 'file_path': file_path, 'type': 'exam_id', 'shape': exam_id[i].shape, 'cache': 0})
