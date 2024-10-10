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

```sh
pip install -r VLM4Building/requirements.txt
```
Detail setup see: [https://huggingface.co/allenai/Molmo-7B-D-0924](https://huggingface.co/allenai/Molmo-72B-0924)

### 3. Download image data
```sh
python3.10 VLM4Building/code/01_get_img_unzip.py
```

### 4. Run prediction (with specify number of GPUs)
```sh
python3.10 VLM4Building/code/02_molmo72b.py --num_gpus 20
```
