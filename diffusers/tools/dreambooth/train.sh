accelerate config default
DATA_NAME="haomo_night"
export MODEL_NAME="/mnt/share_disk/lei/git/diffusers/local_models/stable-diffusion-v1-5"
# export INSTANCE_DIR="./data/train/finetune/$DATA_NAME"
export INSTANCE_DIR="/mnt/ve_share/songyuhao/generation/data/train/GAN/night/trainB"
export OUTPUT_DIR="./res/finetune/dreambooth/${DATA_NAME}_sks_200x"
export CLASS_DIR="./data/train/finetune/night_class"

accelerate launch ./examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks night traffic scene" \
  --class_prompt="a photo of night traffic scene" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_checkpointing \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=32000 \
  --checkpointing_steps=10000

#   --use_8bit_adam \



