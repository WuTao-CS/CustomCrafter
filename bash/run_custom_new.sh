name="epoch99"
prompt_file="prompts/happysad_prompts.txt"

config=configs/custom_train_wrapper_inference2.yaml
res_dir="results/result_custom_diffusion_wrapper_happysad_crossatten"

ckpt='logs/2023-11-11T11-09-44_plushie-happysad-lvdm-wrapper-plush-toy-crossattn/checkpoints/epoch=000099.ckpt'
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=21917 \
    pipeline/evaluation/ddp_wrapper.py \
    --module 'inference' \
    --seed 1000 \
    --ckpt_path $ckpt \
    --base $config \
    --savedir $res_dir/$name \
    --n_samples 1 \
    --bs 1 --height 320 --width 512 \
    --unconditional_guidance_scale 15.0 \
    --ddim_steps 50 \
    --ddim_eta 1.0 \
    --prompt_file $prompt_file