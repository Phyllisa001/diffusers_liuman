accelerate config default
DATA_NAME="haomo_night"
export MODEL_NAME="/mnt/share_disk/lei/git/diffusers/local_models/stable-diffusion-v1-5"
export DATA_DIR="./data/train/finetune/$DATA_NAME"
export OUTPUT_DIR="./res/finetune/textual_inversion/$DATA_NAME"

accelerate launch ./examples/textual_inversion/textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<haomo-night>" --initializer_token="night" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \

  # If inf during training
  # --validation_prompt="A <cat-toy> backpack" \
  # --num_validation_images=4 \
  # --validation_steps=100

  # Checkpoint steps & resume
  # --checkpointing_steps=500
  # --resume_from_checkpoint="checkpoint-1500"