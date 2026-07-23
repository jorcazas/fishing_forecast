from typing import List
import configparser
from datetime import datetime
import os
import csv
from motu_utils import motu_api
import motuclient

class MotuOptions:
    def __init__(self, attrs: dict):
        super(MotuOptions, self).__setattr__("attrs", attrs)

    def __setattr__(self, k, v):
        self.attrs[k] = v

    def __getattr__(self, k):
        try:
            return self.attrs[k]
        except KeyError:
            return None
        
def read_credentials() -> List[str]:
    """
    Reads credentials for copernicus from the config.ini file.

    Returns:
        A list of strings containing the username and password for the MOTU API.

    """
    config = configparser.ConfigParser()
    config.read('config/config.ini')

    username = config.get('api_keys', 'user_cop')
    password = config.get('api_keys', 'password_cop')
    
    return [username, password]

def read_variable_list(file_path):
    '''
    Reads the CSV file with the list of variables to download from the Copernicus API.	
    
    file_path (str): Path to the CSV file with the list of variables to download from the Copernicus API.
    Returns: 
        a list of dictionaries, each one containing the information of a row in the CSV file.
    '''

    # Create an empty list to store the dictionaries
    var_dict_list = []

    # Read the CSV file
    with open(file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        
        # Iterate over each row in the CSV file
        for row in reader:
            # Append the row dictionary to the list
            var_dict_list.append(row)

    # Return the list of dictionaries
    return var_dict_list

def create_request(service_id, product_id, date, motu, directory_to, name, user, password):
    '''
    Creates a request with the parameters needed to download the data from the Copernicus API.
    
    service_id (str): Service ID of the service that manages the product we want to download 
    product_id (str): ID of the product we want to download
    date (str): Initial date from where we want to download the data, in format YYYY-MM-DD
    motu (str): URL to the server where the product is hosted
    directory_to (str): Directory where the downloaded product will be placed 
    name (str): Name under which the downloaded product will be saved
    user (str): User with which we'll connect to the server
    password (str): Password with which we'll connect to the server

    Returns:
        A dictionary with the parameters needed to download the data from the Copernicus API.

    ''' 
    str_date = date.strftime('%Y-%m-%d')
    year = str_date[:4]
    month = str_date[5:7]
    last_month = f'{int(month) - 1:02}' 

    
    return {"service_id": service_id,
            "product_id": product_id,
            "date_min": datetime.strptime(f'{year}-{last_month}-01', '%Y-%m-%d').date(),
            "date_max": date.date(),
            "longitude_min": -117.,
            "longitude_max": -112.5,
            "latitude_min": 28.,
            "latitude_max": 32.,
            "variable": [],
            "motu": motu,
            "out_dir": directory_to,
            "out_name": name+".nc",
            "auth_mode": "cas",
            "user": user,
            "pwd": password
            }
file_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
destination_path=os.path.join(file_path,'etl', 'data', 'copernicus', 'raw', 'last_month')

def make_request(credentials,
                 file_path=file_path, 
                 destination_path=destination_path):
    '''
    Creates a request with the parameters needed to download the data from the Copernicus API.

    credentials (list): List of strings containing the username and password for the MOTU API.
    file_path (str): Path to the CSV file with the list of variables to download from the Copernicus API.
        default: same directory as this file
    destination_path (str): Path to the directory where the downloaded product will be placed
        default: same directory as this file/data/copernicus/raw
    '''

    # Read the CSV file with the list of variables to download from the Copernicus API
    var_dict_list = read_variable_list(os.path.join(file_path, "infra", "copernicus_var_dict.csv"))
    date = datetime.today()

    for dict in var_dict_list:
        try:
            

            # Construct a relative file path to the data directory

            directory_to = destination_path
            if not os.path.isdir(directory_to):
                os.makedirs(directory_to, exist_ok=True)


            data_request_options_dict_manual = create_request(dict["service_id"],
                                                            dict["product_id"],
                                                            date,
                                                            dict["motu"],
                                                            directory_to, 
                                                            dict["name"],
                                                            credentials[0], 
                                                            credentials[1]
                                                            )
        
            motu_api.execute_request(MotuOptions(data_request_options_dict_manual))

        
        except Exception as e:
            print(e)
