import pandas as pd
import os

def main():

    # Get the absolute path to the directory containing the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the absolute path to the parent directory of the script directory
    parent_dir = os.path.dirname(script_dir)

    # Construct a relative file path to the data directory
    directory = os.path.join(parent_dir, 'data', 'globcolour', 'processed')

    combined_df = pd.DataFrame(columns=['date', 'variable', 'value', 'lat', 'lon'])

    for variable in os.listdir(directory):
        variable_directory = os.path.join(directory, variable)

        for filename in os.listdir(variable_directory):

            dataset = pd.read_csv(os.path.join(variable_directory, filename))
        
            for date in dataset['date'].unique():
                # Filter the dataset by date
                subset = dataset[dataset['date'] == date]
                
                # Get the value
                mean_value = subset[[col for col in subset.columns if 'mean' in col][0]]
                
                # Get the coordinates
                lat = subset['lat']
                lon = subset['lon']
                mean_value.to_csv("test.csv")
                # Append the data to the combined DataFrame
                date_df = pd.DataFrame({'date': date,
                                                'variable': f'Variable_{variable}',
                                                'value': mean_value,
                                                'lat': lat,
                                                'lon': lon})
                date_df = date_df.dropna(subset=['value'])
                combined_df = pd.concat([combined_df, date_df], ignore_index=True)
    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(os.path.join(parent_dir, 'data', 'globcolour', 'combined.csv'))


if __name__ == "__main__":
    main()