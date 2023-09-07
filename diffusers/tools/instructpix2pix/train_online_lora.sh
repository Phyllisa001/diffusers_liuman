pip install -e ".[torch]"

helpFunction()
{
   echo ""
   echo "Usage: $0 -d DATASET_ID -o OUTPUT_DIR -t ITER -b BATCH_SIZE"
   echo -e "\t-d Description of what is DATASET_ID"
   echo -e "\t-o Description of what is OUTPUT_DIR"
   echo -e "\t-t Description of what is ITER"
   echo -e "\t-b Description of what is BATCH_SIZE"
   echo -e "\t-r Description of what is RANK"
   echo -e "\t-l Description of what is LR"
   exit 1 # Exit script after printing help
}

while getopts "d:o:t:b:r:l:" opt
do
   case "$opt" in
      d ) DATASET_ID="$OPTARG" ;;
      o ) OUTPUT_DIR="$OPTARG" ;;
      t ) ITER="$OPTARG" ;;
      b ) BATCH_SIZE="$OPTARG" ;;
      r ) RANK="$OPTARG" ;;
      l ) LR="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$DATASET_ID" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$ITER" ] || [ -z "$BATCH_SIZE" ] || [ -z "$RANK" ] || [ -z "$LR" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Begin script in case all parameters are correct
echo "$DATASET_ID"
echo "$OUTPUT_DIR"
echo "$ITER"
echo "$BATCH_SIZE"
echo "$RANK"
echo "$LR"


export MODEL_NAME="/share/songyuhao/generation/models/online/diffusions/base/instruct-pix2pix"

accelerate launch --multi_gpu ./examples/instruct_pix2pix/train_instruct_pix2pix_lora.py \
 --pretrained_model_name_or_path=$MODEL_NAME \
 --use_ema \
 --dataset_name=$DATASET_ID \
 --resolution=512 --random_flip \
 --gradient_accumulation_steps=4 --gradient_checkpointing \
 --max_train_steps=$ITER \
 --train_batch_size=$BATCH_SIZE \
 --checkpointing_steps=100 \
 --learning_rate=$LR --lr_warmup_steps=0 \
 --conditioning_dropout_prob=0.05 \
 --seed=42 \
 --rank=$RANK \
 --output_dir=$OUTPUT_DIR


#   --enable_xformers_memory_efficient_attention \

# pip install safetensors
# python ./scripts/convert_diffusers_to_original_stable_diffusion.py --use_safetensors --model_path $OUTPUT_DIR --checkpoint_path $OUTPUT_DIR/model.safetensors
# cp $OUTPUT_DIR/model.safetensors /cpfs/model/model.safetensors
