# VLM4Building

## Installation

### 0. Clone the Repository

```sh
git clone https://github.com/seshing/VLM4Building.git
```

### 1. Set a virtual envs

```sh
conda create -n molmo python=3.10
conda activate molmo
```

### 2. Install packages
1. Molmo models:
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
