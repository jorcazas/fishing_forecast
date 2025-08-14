import netCDF4 as nc
import xarray as xr
import pandas as pd
from ftplib import FTP
import os 
import re
import numpy as np

from globcolour import read_variable_dict, process_data

def free_storage():
    """
    Frees up storage by clearing files in the raw data directory that are already in the processed data directory.
    Clearing (and not removing) files is done to avoid having to re-download them.
    """
    # Get the absolute path to the directory containing the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the absolute path to the parent directory of the script directory
    parent_dir = os.path.dirname(script_dir)

    # Construct a relative file path to the data directory
    directory = os.path.join(parent_dir, 'data', 'globcolour', 'raw')

    print(os.listdir(directory))
    for variable in os.listdir(directory):
        variable_directory = os.path.join(directory, variable)
        print(os.path.join(parent_dir, 'data', 'globcolour', 'processed', variable))
        for filename in os.listdir(variable_directory):
            # if filename was already processed, clear it so it doesn't take up space
            if filename.replace('.nc', '_clean.csv') in os.listdir(os.path.join(parent_dir, 'data', 'globcolour', 'processed', variable)):
                # clear file
                open(os.path.join(variable_directory, filename), 'w').close()
                


def main():

    free_storage()
    print("Storage freed, proceeding to transform data...")

    # Get the absolute path to the directory containing the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the absolute path to the parent directory of the script directory
    parent_dir = os.path.dirname(script_dir)

    # Construct a relative file path to the data directory
    directory_from = os.path.join(parent_dir, 'data', 'globcolour', 'raw')

    # Construct a relative file path to the data directory
    if not os.path.isdir(os.path.join(parent_dir, 'data', 'globcolour')):
        os.mkdir(os.path.join(parent_dir, 'data', 'globcolour'))
    if not os.path.isdir(os.path.join(parent_dir, 'data', 'globcolour', 'processed')):
        os.mkdir(os.path.join(parent_dir, 'data', 'globcolour', 'processed'))

    directory_to = os.path.join(parent_dir, 'data', 'globcolour', 'processed')

    variable_dict = read_variable_dict('input/variable_dict.csv')

    for v in list(variable_dict.keys()):
        process_data(v, directory_from, directory_to)

if __name__ == '__main__':
    main()
            