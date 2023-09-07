accelerate config default
DATA_NAME="haomo_night"

export MODEL_NAME="/share/liuman/SD-HM/SD-HM-V0.4.0"
# export INSTANCE_DIR="./data/train/finetune/$DATA_NAME"
# export MODEL_NAME="/mnt/ve_share/songyuhao/generation/models/online/diffusions/res/finetune/dreambooth/SD-HM-V0.4.0"
# export INSTANCE_DIR="./data/train/finetune/$DATA_NAME"

# export INSTANCE_DIR="/mnt/share_disk/liuman/data/blip_tgt/imgs"
export INSTANCE_DIR="/share/liuman/parquet/"
# export OUTPUT_DIR="./res/finetune/dreambooth/${DATA_NAME}_sks_200x"
export OUTPUT_DIR="/share/liuman/model/finetune"
# export CLASS_DIR="./data/train/finetune/night_class"
# /mnt/share_disk/liuman/diffusers/examples/text_to_image/train_text_to_image_liu.py
# accelerate launch ./examples/dreambooth/train_dreambooth.py \
accelerate launch --multi_gpu /share/liuman/diffusers/examples/text_to_image/train_text_to_image_liu.py \
  --input_perturbation=0.1 \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --gradient_checkpointing \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=320000 \
  --checkpointing_steps=100000

#   --use_8bit_adam \
