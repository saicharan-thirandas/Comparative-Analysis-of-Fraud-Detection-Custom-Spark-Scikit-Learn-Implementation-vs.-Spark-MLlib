import requests

def download_file(url, destination):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(destination, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

# Example usage
url = "https://www.kaggle.com/datasets/ealaxi/paysim1/download?datasetVersionNumber=2"
destination = "fraud_dataset.csv"

download_file(url, destination)
