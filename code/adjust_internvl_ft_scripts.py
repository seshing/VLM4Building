import json
import os

### After this you can run finetune script: 
# cd models/InternVL/internvl_chat
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 GPUS=5 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl2.0/2nd_finetune/internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_lora_building.sh
# PYTHONPATH="/public/home/zhaotianhong/xiucheng/models/InternVL/internvl_chat" python tools/merge_lora.py /public/home/zhaotianhong/xiucheng/models/InternVL/internvl2_finetune/InternVL2_2b_2e_fullgpt_loraosm /public/home/zhaotianhong/xiucheng/models/InternVL/internvl2_finetune/InternVL2_2b_2e_fullgpt_loraosm_merged


img_dir = "building_dataset/img"
annotations = "building_dataset/jsonl/train_1g1o_1203.jsonl"
repeat_time = 1

### If the path of the data is not in the right format for shell file, you can adjust it by running the following code:
# chdir = 'path/to/your/chdir'
# img_dir = os.path.join(chdir, 'building_dataset/img')
# annotations = os.path.join(chdir, 'building_dataset/jsonl/train_1g1o_1203.jsonl')

def get_length(annotations):
    with open(annotations, 'r') as f:
        length = sum(1 for _ in f)
    return length

data = {
    "osm_caption": {
        "root": img_dir,
        "annotation": annotations,
        "data_augment": False,
        "repeat_time": repeat_time,
        "length": get_length(annotations)
    }
    
}

with open("models/InternVL/internvl_chat/shell/data/internvl_1_2_finetune_building.json", "w") as json_file:
        json.dump(data, json_file, indent=4)
        

# """
# Specify the file name you want to adjust
# For example, "internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_lora.sh",
# "internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_full.sh", 
# "internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_full.sh", 
# "internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora.sh"
# """

mode = 'full'
adjust_file = f'internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_{mode}.sh'
save_file = f'internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_{mode}_building.sh'
sh_dir = 'models/InternVL/internvl_chat/shell/internvl2.0/2nd_finetune'
# sh_dir = os.path.join(chdir, sh_dir)

adjust_train_epochs = 3
adjust_learning_rate = 4e-5 # Defualt 4e-5
adjust_freeze_backbone = False
freeze_mode = 'unfreeze' if not adjust_freeze_backbone else 'freeze'

adjust_OUTPUT_DIR = f'models/InternVL/internvl2_finetune/internvl2_5_8b_{adjust_train_epochs}e_{adjust_learning_rate}_1G1O_{freeze_mode}_{mode}'
adjust_model_path = "models/InternVL/InternVL2_5-8B"
adjust_meta_path = 'models/InternVL/internvl_chat/shell/data/internvl_1_2_finetune_building.json'


with open(os.path.join(sh_dir, adjust_file), 'r') as file:
    filedata = file.read()

filedata = filedata.replace(
    "'work_dirs/internvl_chat_v2_0/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_full'",
    # "'work_dirs/internvl_chat_v2_0/internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_lora'",
    f"'{adjust_OUTPUT_DIR}'"
)

filedata = filedata.replace(
    '--meta_path "./shell/data/internvl_1_2_finetune_custom.json" \\',
    f'--meta_path "{adjust_meta_path}" \\'
)

filedata = filedata.replace(
    '--num_train_epochs 1 \\',
    f'--num_train_epochs {adjust_train_epochs} \\'
)

filedata = filedata.replace(
    '--learning_rate 4e-5 \\',
    f'--learning_rate {adjust_learning_rate} \\'
)

filedata = filedata.replace(
    '--freeze_backbone True \\',
    f'--freeze_backbone {adjust_freeze_backbone} \\'
)

filedata = filedata.replace(
    '--model_name_or_path "./pretrained/InternVL2-2B" \\',
    f'--model_name_or_path "{adjust_model_path}" \\'
)


with open(os.path.join(sh_dir, save_file), 'w') as file:
    file.write(filedata)
    # with open('internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_full_building.sh', 'w') as new_file:
    #     new_file.write(filedata)

