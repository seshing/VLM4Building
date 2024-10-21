import os
import json
import torch
import multiprocessing
import argparse
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

def update_json_file(data, file_path):
    try:
        with open(file_path, 'r') as file:
            existing_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = {}
    
    existing_data.update(data)

    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)

def process_images(gpu_id, image_files, model_name, image_directory, output_directory):
    # Set the GPU
    torch.cuda.set_device(gpu_id)
    
    # Load model and processor
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    
    output_file1 = os.path.join(output_directory, f'responses1_gpu{gpu_id}.json')   
    output_file2 = os.path.join(output_directory, f'responses2_gpu{gpu_id}.json')

    for image_name in tqdm(image_files, desc=f'GPU {gpu_id}'):
        full_image_path = os.path.join(image_directory, image_name)

        # First round: Generate description
        inputs = processor.process(
            images=[Image.open(full_image_path)],
            text="""
            Analyze the building shown in the image and provide a detailed description of its architectural features. Then, describe the building type, the building's age (by specifying an approximate construction year), the primary facade material (the main material visible on the building's surface), the construction material, and the total number of floors in the building.
            """
        )

        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=500, stop_strings="<|endoftext|>"),
                tokenizer=processor.tokenizer
            )   

        generated_tokens = output[0, inputs['input_ids'].size(1):]
        description_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        description_result = {image_name: description_text}
        update_json_file(description_result, output_file1)

        # Second round: Generate JSON summary
        inputs = processor.process(
            images=[Image.open(full_image_path)],
            text=description_text.strip() + """
            Based on the analysis of the building, provide concise labels for each category using the following JSON format. Select appropriate values from the provided options for each category:
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
        )

        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                tokenizer=processor.tokenizer
            )

        generated_tokens = output[0, inputs['input_ids'].size(1):]
        json_output = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        temp_result = {image_name: json_output}
        update_json_file(temp_result, output_file2)

def main():
    parser = argparse.ArgumentParser(description="Process images using multiple GPUs.")
    parser.add_argument('--num_gpus', type=int, default=1, help="Number of GPUs to use")
    parser.add_argument('--model', type=str, default='allenai/Molmo-7B-D-0924', help="VLM model used")
    args = parser.parse_args()

    model_name = args.model
    image_directory = 'code/VLM4Building/img'
    model_output_name = model_name.split('/')[-1]
    output_directory = f'code/VLM4Building/output/{model_output_name}'
    os.makedirs(output_directory, exist_ok=True)

    num_gpus = args.num_gpus

    all_image_files = [img for img in os.listdir(image_directory) if img.endswith((".jpg", ".jpeg", ".png"))]
    
    # Split the image files into groups based on the number of GPUs
    images_per_gpu = len(all_image_files) // num_gpus
    image_files_split = [all_image_files[i:i + images_per_gpu] for i in range(0, len(all_image_files), images_per_gpu)]

    # If there are any remaining images, add them to the last group
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
