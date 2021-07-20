# The name of experiment

export WANDB_API_KEY=9fe5e176e6d0edecf332cdf97a58020645efb6d4

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/pretrain.py \
        --distributed --multiGPU --fp16 \
        --train mscoco_resplit_train,vgnococo \
        --valid mscoco_resplit_val \
        --batch_size 320 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --lr 1e-4 \
        --num_workers 1 \
        --clip_grad_norm 1.0 \
        --backbone 't5-base' \
        ${@:2} \
        --epoch 30 \
        --caption_only \
