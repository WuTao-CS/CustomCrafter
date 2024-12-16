name="epoch99"
prompt_file="prompts/teddybear_prompts.txt"

config=configs/custom_train_wrapper_inference2.yaml
ckpt='logs/2023-10-28T05-12-26_plushie-teddybear-lvdm-wrapper-teddybear-crossattn/checkpoints/epoch=000099.ckpt'

res_dir="results/result_custom_diffusion_wrapper_teddybear_crossatten_timestep_20"
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=21729 \
    pipeline/evaluation/ddp_wrapper.py \
    --module 'timestep_inference' \
    --seed 1000 \
    --ckpt_path $ckpt \
    --pretrain_path 'logs/2023-10-28T05-12-26_plushie-teddybear-lvdm-wrapper-teddybear-crossattn/checkpoints/epoch=000099_kv.ckpt' \
    --base $config \
    --savedir $res_dir/$name \
    --n_samples 1 \
    --bs 1 --height 320 --width 512 \
    --unconditional_guidance_scale 15.0 \
    --ddim_steps 50 \
    --ddim_eta 1.0 \
    --prompt_file $prompt_file \
    --mid_step 20