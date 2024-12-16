
name="epoch99"
ckpt='logs/2023-10-25T23-27-32_plushie-teddybear-lvdm-wrapper-plush-toy-crossattn/checkpoints/epoch=000099.ckpt'
config='logs/2023-10-25T23-27-32_plushie-teddybear-lvdm-wrapper-plush-toy-crossattn/configs/2023-10-25T23-27-32-project.yaml'

prompt_file="prompts/teddybear_prompts.txt"
res_dir="results/result_custom_diffusion_wrapper_teddybear_crossatten_fix_bug"


CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=23416\
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
    --prompt_file $prompt_file > inference_49.log 2>&1 &

name="epoch89"
ckpt='logs/2023-10-25T23-27-32_plushie-teddybear-lvdm-wrapper-plush-toy-crossattn/checkpoints/epoch=000089.ckpt'
CUDA_VISIBLE_DEVICES=1 nohup python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=23426\
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
    --prompt_file $prompt_file > inference_09.log 2>&1 &

name="epoch79"
ckpt='logs/2023-10-25T23-27-32_plushie-teddybear-lvdm-wrapper-plush-toy-crossattn/checkpoints/epoch=000079.ckpt'
CUDA_VISIBLE_DEVICES=2 nohup python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=23436\
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
    --prompt_file $prompt_file > inference_79.log 2>&1 &

name="epoch69"
ckpt='logs/2023-10-25T23-27-32_plushie-teddybear-lvdm-wrapper-plush-toy-crossattn/checkpoints/epoch=000069.ckpt'
CUDA_VISIBLE_DEVICES=3 nohup python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=23466\
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
    --prompt_file $prompt_file > inference_29.log 2>&1 &