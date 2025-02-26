import requests
from tqdm import tqdm
import zipfile
import os
from huggingface_hub import hf_hub_download, snapshot_download

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
urls = [
    ("https://huggingface.co/datasets/seshing/buildingfacade/resolve/main/Berlin.zip", "VLM4Building/Berlin.zip", "VLM4Building/")
    # ("https://huggingface.co/datasets/seshing/buildingfacade/resolve/main/Manila.zip", "VLM4Building/Manila.zip", "VLM4Building/")
]

if __name__ == "__main__":
    for url, output_zip_path, extract_dir in urls:
        # Step 1: Download the file
        download_file(url, output_zip_path)
        # Step 2: Unzip the file
        unzip_file(output_zip_path, extract_dir)
    # # Step 3: Download the model repository
    model_path = snapshot_download(repo_id="seshing/internvl_buildingfacades",
                                 local_dir="VLM4Building/model/internvl2_5-2b-finetune",
                                 token=None)  # Add your token if the repo is private
    print(f"Model downloaded to: {model_path}")
