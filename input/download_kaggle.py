import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Set your Kaggle API credentials
os.environ['KAGGLE_USERNAME'] = 'your_kaggle_username'
os.environ['KAGGLE_KEY'] = 'your_kaggle_key'

# Initialize the Kaggle API
api = KaggleApi()

# Set the dataset URL
dataset_url = 'https://www.kaggle.com/datasets/ealaxi/paysim1'

# Extract the dataset name from the URL
dataset_name = dataset_url.split('/')[-1]

# Create a directory to save the downloaded dataset
os.makedirs(dataset_name, exist_ok=True)

# Change the current working directory to the dataset directory
os.chdir(dataset_name)

# Download the dataset using the Kaggle API
api.dataset_download_files(dataset_url, unzip=True)

# Rename the CSV file to 'fraud_dataset.csv'
for file in os.listdir():
    if file.endswith('.csv'):
        os.rename(file, 'fraud_dataset.csv')

# Change the current working directory back to the original directory
os.chdir('..')