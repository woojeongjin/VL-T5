# The name of experiment

export WANDB_API_KEY=9fe5e176e6d0edecf332cdf97a58020645efb6d4
export NGPU=$1

if [[ "$PATH" == *"vlt5"* ]]; then
  echo "vlt5 env activated"
else
  echo "vlt5 env not activated"
  conda_base=$(conda info --base)
  source ${conda_base}/etc/profile.d/conda.sh
  conda activate vlt5
fi

name=VLT5

output=snap/COCOCaption/$name
python -m spacy download en_core_web_sm 
python -c "import language_evaluation; language_evaluation.download('coco')"


PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/caption.py \
        --distributed --multiGPU \
        --train karpathy_train \
        --valid karpathy_val \
        --test karpathy_test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 20 \
        --num_workers 4 \
        ${@:2} \
        --num_beams 5 \
        --batch_size 80 \
        --valid_batch_size 100 \
        --test_only \