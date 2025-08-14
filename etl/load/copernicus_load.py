import glob
import xarray as xr
import os


input_dir = os.path.join(os.getcwd(),"data","copernicus", "raw", "last_month")
destination_dir = os.path.join(os.getcwd(),"data","copernicus", "processed", "last_month")

if not os.path.isdir(input_dir):
     os.makedirs(input_dir, exist_ok=True)

if not os.path.isdir(destination_dir):
     os.makedirs(destination_dir, exist_ok=True)

def nc_to_csv(input_dir=input_dir, # PENDIENTE: hacer try catch
           destination_dir=destination_dir):
    
    for filename in os.listdir(input_dir):
        raw_file = os.path.join(input_dir, filename)

        ds = xr.open_dataset(raw_file)

        df = ds.to_dataframe()
        df = df.reset_index()
        ds.close()

        if not os.path.isdir(destination_dir):
                os.makedirs(destination_dir, exist_ok=True)

        
        df.to_csv(os.path.join(destination_dir, filename[:-3]+".csv")) # AGREGAR LIMPIEZA FINAL

def main():

    nc_to_csv(input_dir=input_dir,
              destination_dir=destination_dir)

if __name__ == '__main__':
    main()


        
