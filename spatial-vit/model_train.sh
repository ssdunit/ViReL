set -e
cd "$(dirname "$0")"
python deit/main.py \
  --model edd_rope_mixed_deit_small_patch8_LS \
  --num_workers 12 \
  --dataset ILSVRC/imagenet-1k \
  --data-pct 100% \
  --input-size 224 \
  --batch-size 48 \
  --epochs 600 \
  --opt adamw \
  --lr 3e-3 \
  --weight-decay 0.02 \
  --warmup-epochs 5 \
  --drop-path 0.05 \
  --smoothing 0.1 \
  --mixup 0.8 --cutmix 1.0 \
  --aa rand-m9-mstd0.5-inc1 \
  --color-jitter 0.3 \
  --unscale-lr \
  --wandb \
  --wandb-project edd-rope-vit \
  --wandb-run-name small-patch8-10pct \
  --output_dir ./runs/edd_small_patch8_pretrain
