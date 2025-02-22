# The name of experiment
export WANDB_API_KEY=9fe5e176e6d0edecf332cdf97a58020645efb6d4
export NGPU=$1



PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/vqa.py \
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
        --backbone 't5-base' \
        ${@:2} \
        --num_beams 5 \
        --batch_size 80 \
        --valid_batch_size 100 \