import csv
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import configparser
import pandas as pd
from ftplib import FTP
import xarray as xr

def read_variable_dict(filename: str) -> Dict[str, str]:
    """
    Reads the variable dictionary from a CSV file and returns it as a dictionary.
    The CSV file should have a "variable" column and a "file_format" column.

    Args:
        filename: A string indicating the name of the CSV file to be read.

    Returns:
        A dictionary mapping variable names to their corresponding file formats.

    """
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        variable_dict = {row['variable']: row['file_format'] for row in reader}
    return variable_dict


def read_credentials() -> List[str]:
    """
    Reads credentials for globcolour from the config.ini file.

    Returns:
        A list of strings containing the username and password for the FTP server.

    """
    config = configparser.ConfigParser()
    config.read('C:/Users/javi2/Documents/COBI/COBI/etl/config/config.ini')

    username = config.get('api_keys', 'user_gc')
    password = config.get('api_keys', 'password_gc')
    
    return [username, password]

def daterange(start_date: datetime, end_date: datetime):
    """
    Generator that yields dates within a given range.

    Args:
        start_date: A datetime object representing the start date of the range.
        end_date: A datetime object representing the end date of the range.

    Yields:
        A datetime object representing a date within the given range.

    """
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def download_raw(variable: str, directory_to: str, resolution: str, 
                 credentials: List[str], variable_dict: Dict[str, str], date_to_download: str) -> List[Tuple[int, int, int]]:
    
    """
    Downloads data from the GlobColour site using FTP.

    Retrieves daily files (of whole years beginning at from_year) from GlobColour through FTP;
    saves them in a given directory without processing them.
    Credentials are stored in config.ini file.

    Args:
        variable: A string indicating the variable to retrieve.
        directory_to: A string indicating the directory where the files will be stored.
        resolution: A string indicating the resolution of the data to be downloaded.
        credentials: A list of strings containing the username and password for the FTP server.
        variable_dict: A dictionary mapping variable names to their corresponding file formats.
        date_to_download: A string indicating the date of the file to be retrieved, in format YYYYmmdd.


    Returns:
        A list of tuples containing the year, month and day of files that were not able to be retrieved.
        For example:

        [(2022, 01, 31), (2023, 02, 01)] 
        this would mean that it was not posible to retrieve files from 2022/01/31 and from 2023/02/01.

        Since it loops through all day numbers from 01 to 31, there will always be dates not retrieved 
        such as (xxxx,02,30), (xxxx,04,31), etc. 

    """
    user = credentials[0] 
    password = credentials[1] 

    directory_path = os.path.join(directory_to, variable)
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    
    os.chdir(directory_path) #changes the active dir - this is where downloaded files will be saved to
    
    
    filematch = 'L3m_*'+variable_dict[variable].replace("RESOLUTION", resolution)+'.nc' # a match for any file in this case, can be changed or left for user to input
    
    
    year = date_to_download[:4]
    month = date_to_download[4:6]
    day = date_to_download[6:]

    try:                    
        download_dir ="GLOB/merged/day/"+year+"/"+month+"/"+day+"/" #dir i want to download files from, can be changed or left for user input
        ftp = FTP("ftp.hermes.acri.fr")
        ftp.login(user,password)

        ftp.cwd(download_dir)
        for filename in ftp.nlst(filematch): # Loop - looking for matching files
            if not os.path.exists(filename):
                fhandle = open(filename, 'wb')
                print('Getting ' + filename) #for confort sake, shows the file that's being retrieved
                ftp.retrbinary('RETR ' + filename, fhandle.write)
                fhandle.close()
                return filename  + " for variable " + variable + " succesfully retrieved"

        ftp.quit()

        
        

    except Exception as e:
        return "error in variable " + variable + ": " + str(e) + "; " + year+month+day

    

def process_data(variable: str, directory_from: str, directory_to: str, coordinates: List[float] = None) -> None:

    """
    Performs data cleansing on NetCDF files for a given variable obtained from a directory directory_from. 
    It extracts data within a specified latitude and longitude range coordinates and saves the cleaned data in 
    a new directory directory_to with the file extension .csv.

    Args:
        variable (string): the name of the variable to be cleaned
        directory_from (string): the path of the directory containing the original files
        directory_to (string): the path of the directory where the cleaned files will be saved
        coordinates (list, optional): a list of four values that specify the range of coordinates to be considered. The default value is [-116, -113, 26, 29].

    Returns:
        This function does not return any value. It only saves the cleaned files in a new directory.

    """

    if coordinates is None:
        coordinates = [-117, -112.5, 28, 32]
        
    directory_path = os.path.join(directory_to, variable)
    
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    os.chdir(directory_path) #changes the active dir - this is where downloaded files will be saved to
    
        
    directory = os.path.join(directory_from, variable)
    
    try:
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)

            try:
                date_match = re.search('\d\d\d\d\d\d\d\d', filename)
                date = pd.to_datetime(date_match.group()).date()

                ds = xr.open_dataset(f)

                df = ds.to_dataframe()
                df = df.reset_index()
                ds.close()

                df = df[(df.lon > coordinates[0]) & (df.lon < coordinates[1]) &
                  (df.lat > coordinates[2]) & (df.lat < coordinates[3])] # este es el rango de coordenadas en el grid

                # Aquí hacemos selección de columnas, las cuales son únicas para cada VARIABLE
                obs = df #observación de un día

                obs["date"] = obs.apply(lambda x: date, axis=1)

                obs.to_csv(filename.replace(".nc", "_clean.csv"))

            except Exception as e:
                print("Error con:", filename, e) #PENDIENTE: quitar prints
    except Exception as e:
        print(e)