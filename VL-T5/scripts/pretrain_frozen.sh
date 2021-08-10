
export WANDB_API_KEY=9fe5e176e6d0edecf332cdf97a58020645efb6d4

export NGPU=$1

PYTHONPATH=$PYTHONPATH:./src \
python src/pretrain_frozen.py \
        --distributed --multiGPU --fp16 \
        --train mscoco_resplit_train,vgnococo \
        --valid mscoco_resplit_val \
        --batch_size 320 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --lr 3e-4 \
        --num_workers 1 \
        --clip_grad_norm 1.0 \
        --losses 'lm' \
        --backbone 'gpt2' \
        ${@:2} \
        --epoch 30 \
        --caption_only \

# python -m torch.distributed.launch \
#         --nproc_per_node=$1 \
#         src/pretrain_frozen.py \
#         --distributed --multiGPU --fp16 \
#         --train mscoco_resplit_train,vgnococo \
#         --valid mscoco_resplit_val \
#         --batch_size 320 \
#         --optim adamw \
#         --warmup_ratio 0.05 \
#         --lr 3e-4 \
#         --num_workers 1 \
#         --clip_grad_norm 1.0 \
#         --losses 'lm' \
#         --backbone 'gpt2' \
#         ${@:2} \
#         --epoch 30 \
#         --caption_only \

        