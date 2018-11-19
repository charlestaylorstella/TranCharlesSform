PROBLEM=translate_ende_wmt32k
MODEL=transformer
HPARAMS=transformer_base_single_gpu
MARK=$1
GPU_ID=-1

if [ ${GPU_ID} -eq -1 ] ; then 
   PYTHON=/bigstore/hlcm2/tianzhiliang/test/software/anaconda3_4_4_0_pytorch0_4_0/bin/python
   #PYTHON=/bigstore/hlcm2/tianzhiliang/test/software/anaconda3_4_4_0_pytorch0_4_0_tf_cpu/bin/python
else
   PYTHON=python
fi

HOME=$(pwd)
DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64/

function predict() {
# Decode

DECODE_FILE=$DATA_DIR/decode_this.txt
echo "Hello world" >> $DECODE_FILE
echo "Goodbye world" >> $DECODE_FILE
echo -e 'Hallo Welt\nAuf Wiedersehen Welt' > ref-translation.de

BEAM_SIZE=4
ALPHA=0.6

export CUDA_VISIBLE_DEVICES=${GPU_ID}
${PYTHON} t2t_decoder.py \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=translation.en

# See the translations
cat translation.en

# Evaluate the BLEU score
# Note: Report this BLEU score in papers, not the internal approx_bleu metric.
${PYTHON} t2t_bleu.py --translation=translation.en --reference=ref-translation.de
}

predict
