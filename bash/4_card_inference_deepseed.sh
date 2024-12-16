
name="epoch19"
ckpt='logs/2023-10-22T23-19-08_plushie-teddybea-lvdm-wrapper-teddybear-updateall/checkpoints/epoch=000019.ckpt/pytorch_model.bin'
config='configs/custom_train_wrapper.yaml'

prompt_file="prompts/reg_prompt.txt"
res_dir="results/result_custom_diffusion_wrapper_teddybear_updateall_reg_prompt"


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

name="epoch29"
ckpt='logs/2023-10-22T23-19-08_plushie-teddybea-lvdm-wrapper-teddybear-updateall/checkpoints/epoch=000029.ckpt/pytorch_model.bin'
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

name="epoch39"
ckpt='logs/2023-10-22T23-19-08_plushie-teddybea-lvdm-wrapper-teddybear-updateall/checkpoints/epoch=000039.ckpt/pytorch_model.bin'
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

name="epoch49"
ckpt='logs/2023-10-22T23-19-08_plushie-teddybea-lvdm-wrapper-teddybear-updateall/checkpoints/epoch=000049.ckpt/pytorch_model.bin'
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