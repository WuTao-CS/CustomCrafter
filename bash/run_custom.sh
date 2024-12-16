DEVICES=0
PORT=10234
LORA=True
prompt_file="prompts/teddybear_prompts.txt"

config=configs/custom_train_wrapper2_lora_test.yaml
log_name="2024-01-07T03-05-27_plushie-teddybear-lvdm-wrapper-teddybear-plush-toy-crossatten-lora-reg-loracustom-aug"
res_dir="results/$log_name/"

name="epoch=000159_kv"
ckpt="logs/2024-01-07T03-05-27_plushie-teddybear-lvdm-wrapper-teddybear-plush-toy-crossatten-lora-reg-loracustom-aug/checkpoints/epoch=000159_kv.ckpt"

CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=$PORT \
    pipeline/evaluation/ddp_wrapper.py \
    --module 'inference' \
    --seed 1000 \
    --lora $LORA \
    --ckpt_path $ckpt \
    --base $config \
    --savedir $res_dir/$name \
    --n_samples 1 \
    --bs 1 --height 320 --width 512 \
    --unconditional_guidance_scale 15.0 \
    --ddim_steps 50 \
    --ddim_eta 1.0 \
    --prompt_file $prompt_file

name="epoch=000199_kv"
ckpt="logs/2024-01-07T03-05-27_plushie-teddybear-lvdm-wrapper-teddybear-plush-toy-crossatten-lora-reg-loracustom-aug/checkpoints/epoch=000199_kv.ckpt"

CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=$PORT \
    pipeline/evaluation/ddp_wrapper.py \
    --module 'inference' \
    --seed 1000 \
    --lora $LORA \
    --ckpt_path $ckpt \
    --base $config \
    --savedir $res_dir/$name \
    --n_samples 1 \
    --bs 1 --height 320 --width 512 \
    --unconditional_guidance_scale 15.0 \
    --ddim_steps 50 \
    --ddim_eta 1.0 \
    --prompt_file $prompt_file