pip install -e ".[torch]"

helpFunction()
{
   echo ""
   echo "Usage: $0 -i INSTANCE_DIR -p INSTANCE_PROMPT -o OUTPUT_DIR -t ITER -b BATCH_SIZE"
   echo -e "\t-i Description of what is INSTANCE_DIR"
   echo -e "\t-p Description of what is INSTANCE_PROMPT"
   echo -e "\t-o Description of what is OUTPUT_DIR"
   echo -e "\t-t Description of what is ITER"
   echo -e "\t-b Description of what is BATCH_SIZE"
   exit 1 # Exit script after printing help
}

while getopts "i:p:o:t:b:" opt
do
   case "$opt" in
      i ) INSTANCE_DIR="$OPTARG" ;;
      p ) INSTANCE_PROMPT="$OPTARG" ;;
      o ) OUTPUT_DIR="$OPTARG" ;;
      t ) ITER="$OPTARG" ;;
      b ) BATCH_SIZE="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$INSTANCE_DIR" ] || [ -z "$INSTANCE_PROMPT" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$ITER" ] || [ -z "$BATCH_SIZE" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Begin script in case all parameters are correct
echo "$INSTANCE_DIR"
echo "$INSTANCE_PROMPT"
echo "$OUTPUT_DIR"
echo "$ITER"
echo "$BATCH_SIZE"

accelerate config default
export DATA_NAME="haomo"
export  ="/mnt/ve_share/songyuhao/generation/models/online/diffusions/base/stable-diffusion-2-1"
# export INSTANCE_DIR="./data/train/finetune/$DATA_NAME"
# export INSTANCE_DIR="/share/songyuhao/generation/data/train/diffusions/5000/imgs"
# export OUTPUT_DIR="/share/songyuhao/generation/models/online/diffusions/res/finetune/dreambooth/${DATA_NAME}_5000_seg1"
# # export CLASS_DIR="./data/train/finetune/night_class"
# export INSTANCE_PROMPT="/share/songyuhao/generation/data/train/diffusions/5000/pmps_seg_test1"

accelerate launch --multi_gpu ./examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt=$INSTANCE_PROMPT \
  --resolution=512 \
  --train_batch_size=$BATCH_SIZE \
  --gradient_checkpointing \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=$ITER \
  --checkpointing_steps=20000

# 1 for 160  max_train_steps * train_batch_size * gpu
  # --with_prior_preservation --prior_loss_weight=1.0 \

  # --class_prompt="a photo of night traffic scene" \
#   --use_8bit_adam \


#  ./tools/dreambooth/train_one_online.sh -i /share/songyuhao/generation/data/train/diffusions/5000/imgs -p /share/songyuhao/generation/data/train/diffusions/5000/pmps_seg_test1 -o /share/songyuhao/generation/models/online/diffusions/res/finetune/dreambooth/haomo_5000_seg1_ttt
# pip install safetensors
# python ./scripts/convert_diffusers_to_original_stable_diffusion.py --use_safetensors --model_path $OUTPUT_DIR --checkpoint_path $OUTPUT_DIR/model.safetensors
# cp $OUTPUT_DIR/model.safetensors /cpfs/model/model.safetensors

echo "done"