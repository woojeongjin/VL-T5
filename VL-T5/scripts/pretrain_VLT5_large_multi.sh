# The name of experiment

export WANDB_API_KEY=9fe5e176e6d0edecf332cdf97a58020645efb6d4


# if [[ "$PATH" == *"base"* ]]; then
#   echo "conda env activated"
# else
#   echo "conda env not activated"
#   conda_base=$(conda info --base)
#   source ${conda_base}/etc/profile.d/conda.sh
#   conda activate base
# fi

PYTHONPATH=$PYTHONPATH:./src \
# export NCCL_DEBUG=INFO; \
export NGPU=$1
python src/pretrain.py \
        --distributed --multiGPU --fp16 \
        --train mscoco_resplit_train,vgnococo \
        --valid mscoco_resplit_val \
        --batch_size 100 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --lr 1e-4 \
        --num_workers 1 \
        --clip_grad_norm 1.0 \
        --backbone 't5-large' \
        ${@:2} \
        --epoch 30 \
        --caption_only \
