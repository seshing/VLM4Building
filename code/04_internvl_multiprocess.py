import torch
from internvl_process import update_json_file, load_image, dynamic_preprocess, find_closest_aspect_ratio, split_model
import os
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torchvision.transforms as T
import json
import pandas as pd 
import argparse
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
import multiprocessing

def process_images(gpu_id, image_files, model_name, image_directory, output_directory):
    # Load model and tokenizer
    path = os.path.join('models/InternVL', model_name)
    # device_map = split_model(model_name.split('/')[-1])
    model = AutoModel.from_pretrained(
        path, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        use_flash_attn=True, 
        trust_remote_code=True,
        device_map='auto'
        # device_map=device_map
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    output_file1 = os.path.join(output_directory, f'responses1_gpu{gpu_id}.json')   
    output_file2 = os.path.join(output_directory, f'responses2_gpu{gpu_id}.json')
    
    for image_name in tqdm(image_files, desc=f"Processing images on GPU {gpu_id}"):
        full_image_path = os.path.join(image_directory, image_name)
        pixel_values = load_image(full_image_path, max_num=12)
        
        if pixel_values is not None:
            pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
            generation_config = dict(max_new_tokens=1024, do_sample=False)
            
            question = """<image>\n  Analyze the building shown in the image and provide a detailed description of its architectural features. Then, describe the building type, the building's age (by specifying an approximate construction year), the primary facade material (the main material visible on the building's surface), and the total number of floors in the building. """
            response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
            temp_result1 = {image_name: response}
            update_json_file(temp_result1, output_file1)
            
            question = """ Based on the analysis of the building, provide concise labels for each category using the following JSON format. Select appropriate values from the provided options for each category:
				        {
				    "building_type": "(choose one option from: 'apartments', 'house', 'retail', 'office', 'hotel', 'industrial', 'religious', 'education', 'public', 'garage')",
				    "alternate_building_type": "(choose another option from: 'apartments', 'house', 'retail', 'office', 'hotel', 'industrial', 'religious', 'education', 'public', 'garage')",
				    "building_age": "(a 4-digit year indicating the approximate construction date of the building)",
				    "floors": "(a numeric value representing the total number of floors)",
				    "surface_material": "(choose one option from: 'brick', 'wood', 'concrete', 'metal', 'stone', 'glass', 'plaster')",
				    "alternate_surface_material": "(choose another option from: 'brick', 'wood', 'concrete', 'metal', 'stone', 'glass', 'plaster')",
				    "construction_material": "(choose one option from: 'brick', 'wood', 'concrete', 'steel', 'other')",
				    "alternate_construction_material": "(choose another option from: 'brick', 'wood', 'concrete', 'steel', 'other')"
				}
        """
            response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
            
            temp_result2 = {image_name: response}
            update_json_file(temp_result2, output_file2)
        else:
            print(f"Skipping {image_name}, image could not be loaded.")

def main():
    parser = argparse.ArgumentParser(description="Process images using multiple GPUs.")
    parser.add_argument('--num_gpus', type=int, default=1, help="Number of GPUs to use")
    parser.add_argument('--model', type=str, default='OpenGVLab/InternVL2-1B', help="VLM model used")
    args = parser.parse_args()
    
    model_name = args.model
    image_directory = 'building_dataset/img'
    model_output_name = model_name.split('/')[-1]
    output_directory = f'output/{model_output_name}'
    os.makedirs(output_directory, exist_ok=True)

    test_df = pd.read_csv('building_dataset/csv/test.csv')
    test_id = test_df['image_name'].tolist()
    
    num_gpus = args.num_gpus
    all_image_files = [img for img in os.listdir(image_directory) if img.endswith((".jpg", ".jpeg", ".png"))]
    all_image_files = [img for img in all_image_files if img in test_id]
    
    # Check for already processed images
    processed_images = set()
    for filename in os.listdir(output_directory):
        if filename.startswith('responses1_') and filename.endswith('.json'):
            file_path = os.path.join(output_directory, filename)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    processed_images.update(json.load(f).keys())
    
    # Filter out already processed images
    remaining_images = [img for img in all_image_files if img not in processed_images]
    
    # Split the remaining images into two groups
    images_per_gpu = len(remaining_images) // num_gpus
    image_files_split = [remaining_images[i:i + images_per_gpu] for i in range(0, len(remaining_images), images_per_gpu)]
    
    if len(image_files_split) > num_gpus:
        image_files_split[num_gpus-1].extend(image_files_split[num_gpus])
        image_files_split = image_files_split[:num_gpus]

    # Create and start processes
    processes = []
    for i in range(num_gpus):
        p = multiprocessing.Process(target=process_images, args=(i, image_files_split[i], model_name, image_directory, output_directory))
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("All processes completed.")

if __name__ == "__main__":
    main()
 
