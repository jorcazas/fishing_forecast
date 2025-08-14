import pandas as pd
import psycopg2
from typing import Dict, Set
import os
import glob
import csv
import numpy as np

from globcolour import read_variable_dict


# PENDIENTE
for variable in read_variable_dict("variable_dict_globcolour.csv").keys():
    
        try:
            # Get the absolute path to the directory containing the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))

            # Get the absolute path to the parent directory of the script directory
            parent_dir = os.path.dirname(script_dir)

            # Construct a relative file path to the data directory
            dir_path = os.path.join(parent_dir, 'data', 'globcolour', 'processed')

            # Use glob to find all CSV files in the directory
            all_files = glob.glob(os.path.join(dir_path, "*.csv"))

            # Concatenate all CSV files into a single Pandas DataFrame
            df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)


            # create a dictionary to map the old column names to new column names
            new_column_names = {}
            for column in df.columns:
                if "mean" in column:
                    new_column_names[column] = "mean"
                if "error" in column:
                    new_column_names[column] = "error"

            # rename the columns using the dictionary created above
            df.rename(columns=new_column_names, inplace=True)

            if "mean" not in df.columns:
                df["mean"] = np.nan

            if "error" not in df.columns:
                df["error"] = np.nan

            # Display the concatenated DataFrame
            df = df.dropna(subset = ["mean"])[["lat", "lon", "mean", "error", "date"]]


            df.to_csv("merged_data.csv")
            df["variable"] = variable


            # set up a connection to the database
            conn = psycopg2.connect( #PENDIENTE: poner el config
                host="localhost",
                database="cobi",
                user="postgres",
                password="admin"
            )

            # create a database cursor
            cur = conn.cursor()

            # Rename the DataFrame columns to match the PostgreSQL table columns
            df.columns = ["lat", "lon", "mean", "error", "date", "variable"]

            # Iterate over the DataFrame rows and insert them into the PostgreSQL table
            for i, row in df.iterrows():
                cur.execute("INSERT INTO globcolour (lat, lon, mean, error, date, variable) VALUES (%s, %s, %s, %s, %s, %s)",
                            (row["lat"], row["lon"], row["mean"], row["error"], row["date"], row["variable"]))

            # Commit the changes and close the database connection
            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            print(variable + " error:", e)