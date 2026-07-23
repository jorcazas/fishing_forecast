import ee

# Initialize the Earth Engine API
ee.Initialize()

# Example:
collection = ''
description = ''
folder = ''

# Define the region of interest (ROI)
roi = ee.Geometry.Rectangle([-180, -90, 180, 90])

# Define the dataset you want to extract
dataset = ee.ImageCollection(collection)

# Filter the dataset based on your requirements
filtered_dataset = dataset.filterDate('2022-01-01', '2022-12-31').filterBounds(roi)

# Select the specific bands you are interested in
selected_bands = filtered_dataset.select(['chlor_a', 'sst'])

# Get the first image from the filtered dataset
first_image = selected_bands.first()

# Print the image information
print('First image:', first_image)

# Export the image to Google Drive or download it locally
export_task = ee.batch.Export.image.toDrive(
    image=first_image,
    description=description,
    folder=folder,
    scale=1000,
    region=roi
)

# Start the export task
export_task.start()