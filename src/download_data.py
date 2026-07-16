import kaggle
import os

def download_dataset():
    dataset = "mlg-ulb/creditcardfraud"
    destination = "data/raw"
    os.makedirs(destination, exist_ok=True)
    
    kaggle.api.dataset_download_files(dataset, path=destination, unzip=True)
    print(f"Dataset downloaded and extracted to: {destination}")

if __name__ == "__main__":
    download_dataset()