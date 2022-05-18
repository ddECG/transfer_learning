# External libraries
from curses import meta
import pandas as pd
import numpy as np
import ast

# Internal libraries

# Constants

# Function
def clean(args):
    """ Cleans PTB-data. """

    # Open file (Literal eval converst text to dict)
    metadata = pd.read_csv(args.path, index_col='ecg_id', converters={'scp_codes': ast.literal_eval})
    
    # Extend scp codes to clols
    scp_apply = metadata.scp_codes.apply(pd.Series)
    scp_cols = scp_apply.columns
    metadata = pd.concat([metadata.drop(['scp_codes'], axis=1), scp_apply], axis=1)

    # Set TRUE/FALSE on scp_codes columns
    metadata = format_scp_codes(metadata, scp_cols)
    
    # Save metadata
    metadata.to_csv(args.save)

# Subfunctions
def format_scp_codes(data, colnames):
    """ Format SCP code columns with TRUE/FALSE. """
    
    # Change cols
    for column in colnames:
        data[column] = data[column].apply(lambda x: False if np.isnan(x) else True)
    
    # return
    return(data)