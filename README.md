# VLM4Building


## Inference global dataset
### 1. Clone the Repository
```sh
git clone https://github.com/seshing/VLM4Building.git
pip install -r VLM4Building/requirements.txt
```
### 2. Download image data and model
```sh
python VLM4Building/code/01_get_img_unzip_global.py
```
Model:
```sh
python VLM4Building/code/01_get_img_unzip_global.py
```

## Finetune InternVL
### 1. Clone the Repository

```sh
git clone https://github.com/seshing/VLM4Building.git
pip install -r VLM4Building/requirements.txt
```

### 2. Download image data
```sh
python VLM4Building/code/01_get_img_unzip.py
```

### 3. Install packages
InternVL models:
```sh
mkdir VLM4Building/models
cd VLM4Building/models
git clone https://github.com/OpenGVLab/InternVL.git
```

```sh
conda create -n internvl python=3.9 -y
conda activate internvl
```

```sh
pip install -r requirements.txt
pip install flash-attn==2.3.6 --no-build-isolation
pip install -r requirements/clip_benchmark.txt
```

Detail setup see: https://internvl.readthedocs.io/en/latest/get_started/installation.html

### 4. Finetuning
1. download pretained model
```sh
cd VLM4Building/models/InternVL
pip install -U huggingface_hub

# Download OpenGVLab/InternVL2_5-8B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2_5-8B --local-dir InternVL2_5-8B
```

2. Open file in ```code/adjust_internvl_ft_scripts.py``` to adjust and update parameters for finetuning .sh file.

3. Run finetuning (insert the directory of the updated .sh file): 
```sh
cd internvl_chat
GPUS=10 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl2.0/2nd_finetune/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_full_building.sh
```

### 5. Evaluating
1. After the finetuning, we need to copy python files from original ```InternVL2_5-8B``` floder to the finetuned model directory:
```sh
cd VLM4Building
cp models/InternVL/InternVL2_5-8B/*.py /path/to/new/finetuned/model/
```

2. Run evaluation on the test set:
```sh
python3.9 code/04_internvl_multiprocess.py --model internvl2_finetune/path/to/new/finetuned/model --num_gpus 5
```
