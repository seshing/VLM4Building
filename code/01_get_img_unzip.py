import requests
from tqdm import tqdm
import zipfile
import os

def download_file(url, output_path):
    # Stream the download with a progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Use tqdm to show progress
    with open(output_path, 'wb') as file, tqdm(
        desc=output_path,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))

def unzip_file(zip_path, extract_dir):
    # Ensure the extract directory exists
    os.makedirs(extract_dir, exist_ok=True)

    # Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extracted files to {extract_dir}")

# Define the URL, output zip file path, and extraction directory
# url = "https://huggingface.co/datasets/seshing/buildingdataset/resolve/main/img.zip"
url = "https://huggingface.co/datasets/seshing/buildingdataset/resolve/main/img_1018.zip"
output_zip_path = "img.zip"
extract_dir = "code/VLM4Building"

if __name__ == "__main__":
    # Step 1: Download the file
    download_file(url, output_zip_path)

    # Step 2: Unzip the file
    unzip_file(output_zip_path, extract_dir)
