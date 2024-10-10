from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import os
import json
import concurrent.futures

def update_json_file(data, file_path):
    try:
        with open(file_path, 'r') as file:
            existing_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = {}
    existing_data.update(data)
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)

def process_image(image_name, image_directory, output_file, processor, model):
    full_image_path = os.path.join(image_directory, image_name)
    
    # Load image
    image = Image.open(full_image_path)
    
    # First round: Describe the building
    inputs = processor.process(
        images=[image],
        text="""
        Describe this building, including building type/function, an estimated building age (in which year), primary building materials (construction material, surface material), and the total number of floors.
        """
    )
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=500, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    description_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Second round: Generate JSON summary
    inputs = processor.process(
        images=[image],
        text=description_text + """
        Conclude the information into concise labels for each category using the following JSON format:
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
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    json_output = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Save result
    temp_result = {image_name: json_output}
    update_json_file(temp_result, output_file)

def main():
    processor = AutoProcessor.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map={'': 'cuda:0'}
    )
    
    # Load multiple instances of the model
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map={'': 'cuda:0'}
    )

    image_directory = 'code/VLM4Building/img'
    output_directory = 'code/VLM4Building/output/molmo7b'
    os.makedirs(output_directory, exist_ok=True)
    output_file = os.path.join(output_directory, 'molmo1b_responses.json')

    try:
        with open(output_file, 'r') as file:
            processed_images = json.load(file).keys()
    except (FileNotFoundError, json.JSONDecodeError):
        processed_images = []

    # List of images to process
    images_to_process = [img for img in os.listdir(image_directory) if img.endswith(('.jpg', '.jpeg', '.png')) and img not in processed_images]

    # Process images in parallel, distribute images among models
    with concurrent.futures.ThreadPoolExecutor() as executor:  # Use a multiplier to balance GPU usage
        futures = []
        for i, img in enumerate(images_to_process):
            model = models[i % len(models)]  # Round-robin assignment of models
            futures.append(executor.submit(process_image, img, image_directory, output_file, processor, model))

        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            remaining = len(futures) - sum(1 for f in futures if f.done())
            if i % 100 == 0 or remaining == 0:
                print(f"Tasks remaining: {remaining}")

        concurrent.futures.wait(futures)
        print(f"Number of parallel tasks: {len(futures)}")

if __name__ == "__main__":
    main()
