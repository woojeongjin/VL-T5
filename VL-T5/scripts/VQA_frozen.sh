# The name of experiment
name=VLT5

output=snap/vqa/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/vqa_frozen.py \
        --distributed --multiGPU \
        --train train \
        --valid nominival \
        --test minival,nominival \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 20 \
        --num_workers 4 \
        --backbone 'gpt2' \
        --output $output ${@:2} \
        ${@:2} \
        --num_beams 5 \
        --batch_size 80 \
        --valid_batch_size 100 \