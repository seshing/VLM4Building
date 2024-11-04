# VLM4Building

## Installation

### 1. Clone the Repository

```sh
git clone https://github.com/seshing/VLM4Building.git
```

### 2. Install packages
1. Molmo models:
```sh
conda create -n molmo python=3.10
conda activate molmo
```
```sh
pip install -r VLM4Building/requirements.txt
```
Detail setup see: https://huggingface.co/allenai/Molmo-72B-0924

2. InternVL models:
```sh
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

### 3. Download image data
```sh
python3.10 VLM4Building/code/01_get_img_unzip.py
```

### 4. Run prediction (with specify number of GPUs)
1. Molmo models (justify the model name by using ```allenai/Molmo-72B-0924``` or ```allenai/Molmo-7B-D-0924```):
```sh
python3.10 VLM4Building/code/03_molmo_multiprocess.py --model allenai/Molmo-72B-0924 --num_gpus 4
```


2.InternVL models (justify the model name by using ```OpenGVLab/InternVL2-26B``` or ```OpenGVLab/InternVL2-40B```):
```sh
python3.9 VLM4Building/code/03_molmo_multiprocess.py --model OpenGVLab/InternVL2-26B --num_gpus 4
```

\
\
## Finetune InternVL

### 1. Clone the Repository

```sh
git clone https://github.com/seshing/VLM4Building.git
pip install -r VLM4Building/requirements.txt
```

### 2. Download image data
```sh
python3.10 VLM4Building/code/01_get_img_unzip.py
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

### 4. Adjust data path for trainning
1. download pretained model
```sh
cd VLM4Building/models/InternVL/internvl_chat
mkdir pretrained
cd pretrained

# pip install -U huggingface_hub
# Download OpenGVLab/InternVL2-8B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-8B --local-dir InternVL2-8B
```

2. Adjust following settings for .sh file VLM4Building/models/InternVL/internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora.sh:
```sh
OUTPUT_DIR= '../InternVL2-8B-finetune'
--meta_path "../../../data/json/internvl_1_2_finetune_custom.json" \
```

3. Run finetuning: 
```sh
GPUS=2 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl2.0/2nd_finetune/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora.sh
```

