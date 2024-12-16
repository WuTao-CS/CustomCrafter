DEVICES=0
PORT=10234

prompt_file="prompts/test_prompts.txt"
config=configs/inference_t2v_512_v2.yaml
res_dir="new_results_15/test_teddy/"

CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=$PORT \
    pipeline/evaluation/ddp_wrapper.py \
    --module 'base_inference' \
    --seed 42 \
    --base $config \
    --savedir $res_dir/$name \
    --n_samples 1 \
    --bs 1 --height 320 --width 512 \
    --unconditional_guidance_scale 15.0 \
    --ddim_steps 50 \
    --ddim_eta 1.0 \
    --pretrain 'checkpoints/videocrafter2/model.ckpt' \
    --prompt_file $prompt_file

res_dir="new_results_15/test_teddy_1000/"

CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=$PORT \
    pipeline/evaluation/ddp_wrapper.py \
    --module 'base_inference' \
    --seed 1000 \
    --base $config \
    --savedir $res_dir/$name \
    --n_samples 1 \
    --bs 1 --height 320 --width 512 \
    --unconditional_guidance_scale 15.0 \
    --ddim_steps 50 \
    --ddim_eta 1.0 \
    --pretrain 'checkpoints/videocrafter2/model.ckpt' \
    --prompt_file $prompt_file
# prompt_file="prompts/inference.txt"
# config=checkpoints/model_512/model.yaml
# res_dir="new_results_15/lecun_v1/"

# CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch \
#     --nproc_per_node=1 --master_port=$PORT \
#     pipeline/evaluation/ddp_wrapper.py \
#     --module 'base_inference' \
#     --seed 412 \
#     --base $config \
#     --savedir $res_dir/$name \
#     --n_samples 1 \
#     --bs 1 --height 320 --width 512 \
#     --unconditional_guidance_scale 15.0 \
#     --ddim_steps 50 \
#     --ddim_eta 1.0 \
#     --pretrain 'checkpoints/model_512/model-001.ckpt' \
#     --prompt_file $prompt_file