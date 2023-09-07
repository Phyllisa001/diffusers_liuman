accelerate config default
DATA_NAME="haomo_night"
export MODEL_NAME="/mnt/share_disk/lei/git/diffusers/local_models/stable-diffusion-v1-5"
# export MODEL_NAME="/mnt/share_disk/lei/git/diffusers/local_models/instruct-pix2pix"
export INSTANCE_DIR="./data/train/finetune/$DATA_NAME"
export OUTPUT_DIR="./res/finetune/dreambooth_lora/${DATA_NAME}_sks_2"
export CLASS_DIR="./data/train/finetune/night_class"

WANDB_DISABLE_SERVICE=true

accelerate launch ./examples/dreambooth/train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks night traffic scene" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-3 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1500 \
  --seed="0" \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a photo of night traffic scene" \

  # --validation_prompt="A photo of sks dog in a bucket" \
  # --validation_epochs=50 \
  # --push_to_hub