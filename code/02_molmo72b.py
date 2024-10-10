import torch
import os
import json
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from multiprocessing import Process, Queue, Manager
import torch.multiprocessing as mp
from tqdm import tqdm
import argparse

def update_json_file(data, file_path):
    try:
        with open(file_path, 'r') as file:
            existing_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = {}
    existing_data.update(data)
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)
        
def get_processed_images(output_file):
    try:
        with open(output_file, 'r') as file:
            processed_data = json.load(file)
        return set(processed_data.keys())
    except (FileNotFoundError, json.JSONDecodeError):
        return set()

def process_images(gpu_id, image_queue, output_file, progress_dict):
    # Set the GPU device
    torch.cuda.set_device(gpu_id)
    
    # Load the processor and model
    processor = AutoProcessor.from_pretrained(
        'allenai/Molmo-72B-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/Molmo-72B-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    while not image_queue.empty():
        try:
            image_name, full_image_path = image_queue.get(timeout=1)
        except Queue.Empty:
            break

        # First round: Generate description
        inputs = processor.process(
            images=[Image.open(full_image_path)],
            text="""
            Describe this building, including building type, an estimated building age (in which year), primary building materials (construction material, surface material), and the total number of floors.
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

        # Second round: Generate JSON summary
        inputs = processor.process(
            images=[Image.open(full_image_path)],
            text=description_text + """
            Conclude the above information into concise labels for each category using the following JSON format:
            {
                "building_type": (choose one option from 'apartments', 'house','retail', 'office', 'hotel', 'industrial', 'religious', 'education', 'public', 'garage'),
                "building_age": （a numeric value representing a 4-character year),
                "floors": （a numeric number),
                "construction_material": (choose one option from 'concrete', 'brick', 'steel', 'wood', 'other'),
                "surface_material": (if applicable, choose one option from 'tile', 'wood', 'concrete', 'metal', 'stone', 'glass', 'other')
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
        
        update_json_file({image_name: json_output}, output_file)
        progress_dict[gpu_id] += 1

def main(num_gpus):
    image_directory = 'code/VLM4Building/img'
    output_directory = 'code/VLM4Building/output'
    os.makedirs(output_directory, exist_ok=True)
    output_file = os.path.join(output_directory, 'molmo72b_responses.json')

    processed_images = get_processed_images(output_file)

    images_to_process = [img for img in os.listdir(image_directory) 
                     if img.endswith(('.jpg', '.jpeg', '.png')) and img not in processed_images]

    image_queue = Queue()
    total_images = len(images_to_process)
    for image_name in images_to_process:
        full_image_path = os.path.join(image_directory, image_name)
        image_queue.put((image_name, full_image_path))

    # Start processes for each GPU
    manager = Manager()
    progress_dict = manager.dict()
    processes = []
    num_gpus = num_gpus  # Specify the number of GPUs to use
    for gpu_id in range(num_gpus):
        progress_dict[gpu_id] = 0
        p = Process(target=process_images, args=(gpu_id, image_queue, output_file, progress_dict))
        p.start()
        processes.append(p)

    with tqdm(total=total_images, desc="Processing Images") as pbar:
        finished = False
        while not finished:
            finished = all(not p.is_alive() for p in processes)
            total_processed = sum(progress_dict.values())
            pbar.n = total_processed
            pbar.refresh()

    # Wait for all processes to complete
    for p in processes:
        p.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process images using multiple GPUs")
    parser.add_argument('--num_gpus', type=int, default=2, help="Number of GPUs to use")
    args = parser.parse_args()
    
    mp.set_start_method('spawn')
    main(num_gpus)


# from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
# from PIL import Image
# import os
# import json
# import concurrent.futures

# def update_json_file(data, file_path):
#     try:
#         with open(file_path, 'r') as file:
#             existing_data = json.load(file)
#     except (FileNotFoundError, json.JSONDecodeError):
#         existing_data = {}
#     existing_data.update(data)
#     with open(file_path, 'w') as file:
#         json.dump(existing_data, file, indent=4)

# def process_image(image_name, image_directory, output_file, processor, model):
#     full_image_path = os.path.join(image_directory, image_name)
    
#     # Load image
#     image = Image.open(full_image_path)
    
#     # First round: Describe the building
#     inputs = processor.process(
#         images=[image],
#         text="""
#         Describe this building, including building type/function, an estimated building age (in which year), primary building materials (construction material, surface material), and the total number of floors.
#         """
#     )
#     inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
#     output = model.generate_from_batch(
#         inputs,
#         GenerationConfig(max_new_tokens=500, stop_strings="<|endoftext|>"),
#         tokenizer=processor.tokenizer
#     )
#     generated_tokens = output[0, inputs['input_ids'].size(1):]
#     description_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

#     # Second round: Generate JSON summary
#     inputs = processor.process(
#         images=[image],
#         text=description_text + """
#         Conclude the information into concise labels for each category using the following JSON format:
#         {
#             "building_function": (choose one option from: 'retail', 'transportation', 'office', 'education', 'hotel', 'roof', 'religious', 'apartments', 'industrial', 'house', 'hospital', 'other civic buildings', 'utility', 'garage'),
#             "construction_material": (choose one option from: 'concrete', 'brick', 'steel', 'wood', 'other'),
#             "surface_material": (choose one option from: 'tile', 'wood', 'concrete', 'metal', 'stone', 'glass', 'other'), if applicable,
#             "building_age": (numeric value representing a 4-character year),
#             "floors": (a numeric number)
#         }
#         """
#     )
#     inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
#     output = model.generate_from_batch(
#         inputs,
#         GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
#         tokenizer=processor.tokenizer
#     )
#     generated_tokens = output[0, inputs['input_ids'].size(1):]
#     json_output = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

#     # Save result
#     temp_result = {image_name: json_output}
#     update_json_file(temp_result, output_file)

# def main():
#     processor = AutoProcessor.from_pretrained(
#         'allenai/Molmo-72B-0924',
#         trust_remote_code=True,
#         torch_dtype='auto',
#         device_map={'': 'cuda:0'}
#     )
    
#     # Load multiple instances of the model
#     model_1 = AutoModelForCausalLM.from_pretrained(
#         'allenai/Molmo-72B-0924',
#         trust_remote_code=True,
#         torch_dtype='auto',
#         device_map={'': 'cuda:0'}
#     )
    
#     model_2 = AutoModelForCausalLM.from_pretrained(
#         'allenai/Molmo-72B-0924',
#         trust_remote_code=True,
#         torch_dtype='auto',
#         device_map={'': 'cuda:1'}
#     )
    
#     models = [model_1, model_2]  # Add more models if needed

#     image_directory = 'code/VLM4Building/img'
#     output_directory = 'code/VLM4Building/output'
#     os.makedirs(output_directory, exist_ok=True)
#     output_file = os.path.join(output_directory, 'molmo72b_responses.json')

#     try:
#         with open(output_file, 'r') as file:
#             processed_images = json.load(file).keys()
#     except (FileNotFoundError, json.JSONDecodeError):
#         processed_images = []

#     # List of images to process
#     images_to_process = [img for img in os.listdir(image_directory) if img.endswith(('.jpg', '.jpeg', '.png')) and img not in processed_images]

#     # Process images in parallel, distribute images among models
#     with concurrent.futures.ThreadPoolExecutor(max_workers=len(models) * 2) as executor:  # Use a multiplier to balance GPU usage
#         futures = []
#         for i, img in enumerate(images_to_process):
#             model = models[i % len(models)]  # Round-robin assignment of models
#             futures.append(executor.submit(process_image, img, image_directory, output_file, processor, model))

#         for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
#             remaining = len(futures) - sum(1 for f in futures if f.done())
#             if i % 100 == 0 or remaining == 0:
#                 print(f"Tasks remaining: {remaining}")

#         concurrent.futures.wait(futures)
#         print(f"Number of parallel tasks: {len(futures)}")

# if __name__ == "__main__":
#     main()
