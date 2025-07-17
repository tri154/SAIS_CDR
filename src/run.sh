DATASET="cdr"
SAVE_PATH="best.pt"
LOG_PATH="log.txt"
SEED=2004
TQDM=True

NUM_EPOCH=30
BATCH_SIZE=4
UPDATE_FREQ=1
WARMUP_RATIO=0.06
MAX_GRAD_NORM=1.0

NEW_LR=1e-4
PRETRAINED_LR=5e-5
ADAM_EPSILON=1e-6

DEVICE="cuda:0"
TRANSFORMER="bert-base-cased"
TYPE_DIM=20
GRAPH_LAYERS=4 #NOTE min=2

LOWER_TEMP=2.0
UPPER_TEMP=20.0
LOSS_TRADEOFF=1.0

python main.py \
  --dataset $DATASET \
  --save_path $SAVE_PATH \
  --log_path $LOG_PATH \
  --seed $SEED \
  --tqdm $TQDM \
  --num_epoch $NUM_EPOCH \
  --batch_size $BATCH_SIZE \
  --update_freq $UPDATE_FREQ \
  --warmup_ratio $WARMUP_RATIO \
  --max_grad_norm $MAX_GRAD_NORM \
  --new_lr $NEW_LR \
  --pretrained_lr $PRETRAINED_LR \
  --adam_epsilon $ADAM_EPSILON \
  --device $DEVICE \
  --transformer $TRANSFORMER \
  --type_dim $TYPE_DIM \
  --graph_layers $GRAPH_LAYERS \
  --lower_temp $LOWER_TEMP \
  --upper_temp $UPPER_TEMP \
  --loss_tradeoff $LOSS_TRADEOFF
