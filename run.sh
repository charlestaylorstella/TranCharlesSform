PROBLEM=translate_ende_wmt32k
MODEL=transformer
HPARAMS=transformer_base_single_gpu
GPU_ID=2

MARK=$1

HOME=$(pwd)
DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS
Result=res/
Log=log/

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR ${Log}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64/

function train() {
# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
export CUDA_VISIBLE_DEVICES=${GPU_ID}
python t2t_trainer.py \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams='batch_size=1024' \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR > ${Log}/log${MARK} 2> ${Log}/err${MARK}
}

function getdata() {
# Generate data
python t2t_datagen.py \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM
}

#getdata
train
