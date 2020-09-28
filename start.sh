NUM_GPUS=$1
SWEEP_ID=$2

echo "NUM GPUS: $NUM_GPUS"
echo "SWEEP_ID: $SWEEP_ID"

for i in $(seq 0 $(($NUM_GPUS-1)));
do
  CUDA_VISIBLE_DEVICES=$i wandb agent $SWEEP_ID &
done
