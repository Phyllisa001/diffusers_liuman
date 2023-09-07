export MODEL_NAME="/mnt/ve_share/songyuhao/generation/models/online/diffusions/base/instruct-pix2pix"
export DATASET_ID="/mnt/ve_share/songyuhao/generation/data/train/diffusions/parquet/replace_blend_reweight_snowy_0.80_0.80_2.00_1000"
export OUTPUT_DIR="/mnt/ve_share/songyuhao/generation/models/online/diffusions/res/instruct-pix2pix-test"

accelerate launch --mixed_precision="fp16" --multi_gpu ./examples/instruct_pix2pix/train_instruct_pix2pix.py \
 --pretrained_model_name_or_path=$MODEL_NAME \
 --dataset_name=$DATASET_ID \
 --use_ema \
 --resolution=512 --random_flip \
 --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing \
 --max_train_steps=15000 \
 --checkpointing_steps=5000 --checkpoints_total_limit=1 \
 --learning_rate=5e-05 --lr_warmup_steps=0 \
 --conditioning_dropout_prob=0.05 \
 --mixed_precision=fp16 \
 --seed=42 \
 --output_dir=$OUTPUT_DIR


#   --enable_xformers_memory_efficient_attention \