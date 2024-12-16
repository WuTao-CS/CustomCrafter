
name="baseline_new"

ckpt='checkpoints/model_512/model-001.ckpt'
config='checkpoints/model_512/model.yaml'
# ckpt='checkpoints/model.ckpt'
# config='configs/inference_t2v_512_v1.yaml'

prompt_file="prompts/inference.txt"
res_dir="results/p2p_test"

use_ddp=1
local_debug=1

if [ $use_ddp == 0 ]; then
python3 pipeline/evaluation/inference.py \
--seed 1024 \
--ckpt_path $ckpt \
--base $config \
--savedir $res_dir/$name \
--n_samples 3 \
--bs 1 --height 320 --width 512 \
--unconditional_guidance_scale 15.0 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_file $prompt_file

fi

if [ $local_debug == 1 ]; then
HOST_GPU_NUM=1
HOST_NUM=1
CHIEF_IP=127.0.0.1
INDEX=0
fi

if [ $use_ddp == 1 ]; then
python3 -m torch.distributed.launch \
--nproc_per_node=$HOST_GPU_NUM --nnodes=$HOST_NUM --master_addr=$CHIEF_IP --master_port=23466 --node_rank=$INDEX \
pipeline/evaluation/ddp_wrapper.py \
--module 'inference' \
--seed 1000 \
--ckpt_path $ckpt \
--base $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 2 --height 320 --width 512 \
--unconditional_guidance_scale 15.0 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_file $prompt_file

fi