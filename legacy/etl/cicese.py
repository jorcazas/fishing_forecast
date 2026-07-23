import pandas as pd
import requests
import os
import glob
from datetime import datetime

# PENDIENTE: formato
def gather_cicese_data(year_from, directory_to, location="isla_cedros"):
    location_dict = {"isla_cedros":"ICDN", "guerrero_negro":"GRON"}
    
    # Column names obtained from CICESE files metadata. None of this files have a header
    columns=["anio","mes","dia","hora","minuto","segundo",
             "id_estacion","voltaje_sistema","nivel_mar_leveltrol","nivel_mar_burbujeador",
             "sw_1","sw_2","temperatura_agua","nivel_mar_ott_rsl", "radiacion_solar",
             "direccion_viento", "magnitud_viento", "temperatura_aire","humedad_relativa",
             "presion_atmosferica","precipitacion","voltaje_estacion_met","nivel_mar_sutron"]

    # df is the dataframe that will allocate all the data
    df = pd.DataFrame()
    
    # We set the directory where we will download the data
    if not os.path.isdir(directory_to):
        os.mkdir(directory_to)
    
    directory_path = os.path.join(directory_to, location)
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)

    os.chdir(directory_path) #changes the active dir - this is where downloaded files will be saved to
    
    # We have data from 2011 to 2021. 
    years = list(range(year_from, datetime.now().year+1))
    for year in years:
    
        # Define the URL of the directory containing the .dat files
        url = "http://redmar.cicese.mx/emmc/DATA/"+location_dict[location]+"/MIN/"+str(year)+"/"

        # Send a GET request to the URL
        response = requests.get(url)

        # Extract the HTML content of the response
        html_content = response.content.decode('utf-8')

        # Find all the .dat file names in the HTML content
        dat_files = []
        for line in html_content.split('\n'):
            if '.dat' in line:
                filename = line.split('href="')[1][:15]
                dat_files.append(filename)
    

        # Download each .dat file and save it in the data directory
        for filename in dat_files:
            try:
                file_url = url + filename
                file_path = os.path.join(directory_path, filename)
                response = requests.get(file_url)
                
                if not os.path.exists(file_path):
                    with open(file_path, 'wb') as f:
                        f.write(response.content)


                    # Open the downloaded file and read its content
                    with open(file_path, 'r') as f:
                        content = f.read()

            except Exception as e:
                print(filename, "no se agregó por: ", e)
                
        # Rename df columns with the ones defined before
        dict_columns = {}
        for col, i in zip(columns, range(len(columns))):
            dict_columns[i] = col
        dict_columns
        df = df.rename(columns=dict_columns)

        # Export csv
        df.to_csv(str(year_from)+"_"+location+".csv")

def read_cicese_data(place, directory_from):
    columns=["anio","mes","dia","hora","minuto","segundo",
             "id_estacion","voltaje_sistema","nivel_mar_leveltrol","nivel_mar_burbujeador",
             "sw_1","sw_2","temperatura_agua","nivel_mar_ott_rsl", "radiacion_solar",
             "direccion_viento", "magnitud_viento", "temperatura_aire","humedad_relativa",
             "presion_atmosferica","precipitacion","voltaje_estacion_met","nivel_mar_sutron"]
    # Set the directory path
    dir_path = directory_from+place
    
    # Get a list of all .dat files in the directory
    dat_files = glob.glob(os.path.join(dir_path, "*.dat"))

    # Initialize an empty list to store the dataframes
    dfs = []

    # Loop through each file and read it into a dataframe
    for file in dat_files:
        df = pd.read_csv(file, lineterminator='\n', delim_whitespace=True, header=None)
        dfs.append(df)

    # Concatenate all the dataframes into a single dataframe
    result_df = pd.concat(dfs, axis=0, ignore_index=True)
    
    # Rename df columns with the ones defined before
    dict_columns = {}
    for col, i in zip(columns, range(len(columns))):
        dict_columns[i] = col
    dict_columns
    result_df = result_df.rename(columns=dict_columns)

    
    return result_df