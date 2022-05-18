# Transfer learning
Transfer learning model from lima_deep_2021 and CODE-15% to ptb_xl.

# Functions
There are several functionalities, including 1) converting wfdb files to hdf5, 2) making predictions of age based on predefined model, 3) training model based on ECG data. To view different functions use:

### Main help
`python main.py --help`

### Conversion help
`python main.py convert --help`

### Prediction help
`python main.py predict --help`

### Training help
`python main.py train --help`

### Convert wfdb to hdf5
Converts wfdb files to hdf5 files. 

**Main function**
`python main.py convert --data DATA_PATH --metadata METADATA_FILE --save_hdf SAVE_HDF_PATH --save_csv SAVE_CSV_PATH`

**Example**
`python main.py convert --data "data-raw/ptb-xl/data/00000" --metadata "data-raw/ptb-xl/metadata.csv" --save_hdf "data/ptbxl.hdf5" --recursive True --replace True --save_csv "data/metadata.csv"`

### Predict age based on ECG data
Create predictions of age based on predefined model.
**Main function**
`python main.py predict --data DATA_PATH --save PREDICTION_SAVE_PATH`

**Example**
`python main.py predict --data "data/ptbxl.hdf5" --save "data/predictions.csv" --replace True`

### Train model based on ECG data
Trains model based on predefined model arcitechture. Can be created by fine tuning or from scratch.

**Main function**
`python main.py train --data DATA_PATH --metadata METADATA_PATH --model MODEL_PATH`
**Example**
`python main.py train --data "data/ptbxl.hdf5" --metadata "data/metadata.csv" --model "model_ptbxl" --replace True --validation_percentage 0.2 --batch_size 6 --epochs 2 --tune True --tune_model "model"`
# TODO
Put in instructions and description of ptb-xl

# Citations
Model and study citation:
Lima, E.M., Ribeiro, A.H., Paixão, G.M.M. et al. Deep neural network-estimated electrocardiographic age as a mortality predictor. Nat Commun 12, 5117 (2021). https://doi.org/10.1038/s41467-021-25351-7. 
Read more at code base https://github.com/antonior92/ecg-age-prediction