export CUBLAS_WORKSPACE_CONFIG=:4096:8
DATASET="cdr"
SAVE_PATH="best.pt"
LOG_PATH="log.txt"
SEED=2004

NUM_EPOCH=30
BATCH_SIZE=4
UPDATE_FREQ=1
WARMUP_RATIO=0.06
MAX_GRAD_NORM=1.0

NEW_LR=1e-4
PRETRAINED_LR=1.472039003976042e-05
ADAM_EPSILON=1e-6

DEVICE="cuda:0"
TRANSFORMER="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
TYPE_DIM=20
GRAPH_LAYERS=3 #NOTE min=2

USE_PSD=True
LOWER_TEMP=2.0
UPPER_TEMP=20.0
LOSS_TRADEOFF=1.0

USE_SC=True
SC_TEMP=0.16096448806072833
SC_WEIGHT=0.1509642367395748

python main.py \
  --dataset $DATASET \
  --save_path $SAVE_PATH \
  --log_path $LOG_PATH \
  --seed $SEED \
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
  --use_psd $USE_PSD \
  --lower_temp $LOWER_TEMP \
  --upper_temp $UPPER_TEMP \
  --loss_tradeoff $LOSS_TRADEOFF \
  --use_sc $USE_SC \
  --sc_temp $SC_TEMP \
  --sc_weight $SC_WEIGHT
