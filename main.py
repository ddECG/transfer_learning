# External imports
import argparse
import sys
from tqdm import tqdm

# Internal imports
from convert_hdf import convert
from predict import predict
from train import train
from clean import clean

# Constants
import constants

# Function
if __name__ == "__main__":

    # Define parsers
    parser = argparse.ArgumentParser(prog='PROG')
    subparsers = parser.add_subparsers(dest='function')

    # Clean arguments
    parser_clean = subparsers.add_parser('clean', help='Clean PTB-xl metadata.')
    parser_clean.add_argument('--path', required=True, type=str,
                            help='Path to metadata (.csv).')                  
    parser_clean.add_argument('--save', required=True, type=str,
                            help='Filename/path to save cleaned data.')

    # Convert arguments
    parser_convert = subparsers.add_parser('convert', help='Convert wfdb files to HDF5.')
    
    parser_convert.add_argument('--data', required=True, type=str,
                            help='Path to data (.hea + .dat).')
    parser_convert.add_argument('--metadata', required=True, type=str,
                            help='Path to metadata (.csv).')                  
    parser_convert.add_argument('--save_hdf', required=True, type=str,
                            help='Path to save converted wfdb (.hdf5).')
    parser_convert.add_argument('--recursive', default=False, type=bool,
                            help='Search single dir or recursivily (default: False).')
    parser_convert.add_argument('--replace', default=False, type=bool,
                            help='Replace exising data if it exists (default: False).')
    parser_convert.add_argument('--id_col', default='ecg_id', type=str,
                            help='Column name for ID in csv file (default: "ecg_id").')
    parser_convert.add_argument('--age_col', default='age', type=str,
                            help='Column name for age inside metadata (default: "age").')
    parser_convert.add_argument('--filename_col', default='filename_hr', type=str,
                            help='Column name for filename inside metadata (default: "filename_hr").')
    parser_convert.add_argument('--save_csv', required=True, type=str,
                            help='Path to save metadata (.csv).')
    parser_convert.add_argument('--folder', default=False, type=str,
                                help='Folders or single file (default: False)')

    # Predict arguments
    parser_predict = subparsers.add_parser('predict', help='Predict using defined model.')
    
    parser_predict.add_argument('--data', required=True, type=str,
                            help='Path to hdf5 containing ECG traces.')
    parser_predict.add_argument('--save', required=True, type=str,
                            help='Path to save predicted data (.csv).')
    parser_predict.add_argument('--model', type=str, default='model',
                            help='Folder containing model (default: "model").')

    parser_predict.add_argument('--traces_dset', type=str, default='tracings', 
                        help='Column name for traces inside datafile (default: "tracings")')
    parser_predict.add_argument('--ids_dset', type=str, default='exam_id', 
                        help='Column name for ID inside datafile (default: "exam_id")')
    parser_predict.add_argument('--age_dset', type=str, default='true_age', 
                        help='Column name for true age inside datafile (default: "true_age")')

    parser_predict.add_argument('--seed', type=int, default='2',
                            help='Random seed used for number generator (default: 2).')
    parser_predict.add_argument('--batch_size', type=int, default='32', 
                        help='Number of exams per hdf5 batch (default: 32).')
    parser_predict.add_argument('--replace', default=False, type=bool,
                            help='Replace exising predictions if it exists (default: False).')
    
    # Train arguments
    parser_train = subparsers.add_parser('train', help='Train model based on data.')
    
    parser_train.add_argument('--data', required=True, type=str,
                            help='Path to hdf5 containing ECG traces.')
    parser_train.add_argument('--metadata', required=True, type=str,
                            help='Path to metadata (.csv).')    
    parser_train.add_argument('--model', type=str, default='model',
                            help='Folder to save model in (default: "model").')
    parser_train.add_argument('--tune', type=bool, default=False,
                            help='Fine tune based on old model. Needs --tune_model if True. (default: "False").')
    parser_train.add_argument('--tune_model', type=str, required='--tune' in sys.argv,
                            help='Folder to old model in. Only required if fine tuning')

    parser_train.add_argument('--id_col', default='ecg_id', type=str,
                            help='Column name for ID inside metadata (default: "ecg_id").')
    parser_train.add_argument('--age_col', default='age', type=str,
                            help='Column name for age inside metadata (default: "age").')
    parser_train.add_argument('--traces_dset', type=str, default='tracings', 
                        help='Column name for traces inside datafile (default: "tracings")')
    parser_train.add_argument('--ids_dset', type=str, default='exam_id', 
                        help='Column name for ID inside datafile (default: "exam_id")')

    parser_train.add_argument('--replace', default=False, type=bool,
                            help='Replace exising model if it exists (default: False).')
    parser_train.add_argument('--validation_percentage', type=float, default=0.2,
                        help='The first `validation_percentage` exams from the data will be for validation (min < .01; max > 0.99).')
    parser_train.add_argument('--batch_size', type=int, default='32', 
                        help='Number of exams per hdf5 batch (default: 32).')
    
    parser_train.add_argument('--seq_length', type=int, default=4096,
                        help='Size (in # of samples) for all traces. If needed traces will be zeropadded to fit into the given size. (default: 4096)')
    parser_train.add_argument('--net_seq_length', type=int, nargs='+', default=[4096, 1024, 256, 64, 16],
                        help='Number of samples per resnet layer (default: [4096, 1024, 256, 64, 16]).')
    parser_train.add_argument('--net_filter_size', type=int, nargs='+', default=[64, 128, 196, 256, 320],
                        help='Filter size in resnet layers (default: [64, 128, 196, 256, 320]).')
    parser_train.add_argument('--kernel_size', type=int, default=17,
                        help='Kernel size in convolutional layers (default: 17).')
    parser_train.add_argument('--dropout_rate', type=float, default=0.8,
                        help='Dropout rate (default: 0.8).')

    parser_train.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')

    parser_train.add_argument("--patience", type=int, default=7,
                        help='Maximum number of epochs without reducing the learning rate (default: 7)')
    parser_train.add_argument("--lr_factor", type=float, default=0.1,
                        help='Reducing factor for the lr in a plateu (default: 0.1)')
    parser_train.add_argument("--min_lr", type=float, default=1e-7,
                        help='Minimum learning rate (default: 1e-7)')

    parser_train.add_argument('--epochs', type=int, default=70,
                        help='Maximum number of epochs (default: 70).')

    # Parse args
    args = parser.parse_args()

    # Run function
    if args.function == 'clean':

        # Info
        tqdm.write("\n\nCleaning PTB-xl metadata. Main settings:")
        tqdm.write(f"\tPath to original file: '{args.path}'")
        tqdm.write(f"\tSave locations: '{args.save}'\n")

        # Run script
        clean(args)
    
    elif args.function == 'convert':

        # Info
        tqdm.write("\n\nConverting WFDB files to hdf5. Main settings:")
        tqdm.write(f"\tWFDB locaiton: '{args.data}'")
        tqdm.write(f"\tMetadata locaiton: '{args.metadata}'")
        tqdm.write(f"\tSave locations: '{args.save_hdf}'\n")

        # Run script
        convert(args)

    if args.function == 'predict':

        # Info
        tqdm.write("\n\nPredicting age. Main settings:")
        tqdm.write(f"\tData file/folder: '{args.data}'")
        tqdm.write(f"\tModel folder: '{args.model}'")
        tqdm.write(f"\tSave folder: '{args.save}'\n\n")

        # Run script
        predict(args)
    
    if args.function == 'train':

        # Info
        tqdm.write("\n\tTraining model. Main settings:")
        tqdm.write(f"\tData file/folder: '{args.data}'")
        tqdm.write(f"\tMetadata: '{args.metadata}'")
        tqdm.write(f"\tSaveing model in: '{args.model}'\n\n")

        # Run script
        train(args)
