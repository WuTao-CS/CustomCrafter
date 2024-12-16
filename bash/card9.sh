DEVICES=0
PORT=14234


LORA=True
MID_STEP=5
BEGIN_SCALE=0.4
MID_SCALE=0.8

config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml


prompt_file="prompts/guitar_prompts.txt"
log_name="2024-03-17T12-35-45_instrument_music3-lvdm-wrapper-guitar-crossatten-lr-0.00003-videocrafter2"
res_dir="rebuttal/step_k/k_5/$log_name"
ckpt="logs/$log_name/checkpoints/epoch=000179.ckpt"
echo $ckpt
CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=$PORT \
    pipeline/evaluation/ddp_wrapper.py \
    --module 'timestep_inference_lora' \
    --seed 1000 \
    --lora $LORA \
    --ckpt_path $ckpt \
    --base $config \
    --savedir $res_dir \
    --n_samples 1 \
    --bs 1 --height 320 --width 512 \
    --unconditional_guidance_scale 12.0 \
    --ddim_steps 50 \
    --ddim_eta 1.0 \
    --pretrain 'checkpoints/videocrafter2/model.ckpt' \
    --prompt_file $prompt_file \
    --mid_step $MID_STEP \
    --begin_scale $BEGIN_SCALE \
    --mid_scale $MID_SCALE 

CUDA_VISIBLE_DEVICES=$DEVICES python eval/eval_clip_id.py \
    --video_dir "$res_dir/samples/" \
    --text_dir "$res_dir/input/" \
    --id_img_dir "datasets/benchmark_dataset/instrument_music3"



prompt_file="prompts/teddybear_prompts.txt"
log_name="2024-01-29T11-14-21_plushie-teddybear-lvdm-wrapper-teddybear-plush-toy-crossatten-onlylora-init-ffn-basic-lr-0.00003-seg-videocrafter2"
res_dir="rebuttal/step_k/k_5/$log_name"
ckpt="logs/$log_name/checkpoints/epoch=000179.ckpt"
echo $ckpt
CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=$PORT \
    pipeline/evaluation/ddp_wrapper.py \
    --module 'timestep_inference_lora' \
    --seed 1000 \
    --lora $LORA \
    --ckpt_path $ckpt \
    --base $config \
    --savedir $res_dir \
    --n_samples 1 \
    --bs 1 --height 320 --width 512 \
    --unconditional_guidance_scale 12.0 \
    --ddim_steps 50 \
    --ddim_eta 1.0 \
    --pretrain 'checkpoints/videocrafter2/model.ckpt' \
    --prompt_file $prompt_file \
    --mid_step $MID_STEP \
    --begin_scale $BEGIN_SCALE \
    --mid_scale $MID_SCALE 

CUDA_VISIBLE_DEVICES=$DEVICES python eval/eval_clip_id.py \
    --video_dir "$res_dir/samples/" \
    --text_dir "$res_dir/input/" \
    --id_img_dir "datasets/benchmark_dataset/plushie_teddybear"

prompt_file="prompts/sunglasses_prompts.txt"
log_name="2024-03-18T08-53-25_sunglasses2-lvdm-wrapper-sunglasses-crossatten-lr-0.00003-videocrafter2"
res_dir="rebuttal/step_k/k_5/$log_name"
ckpt="logs/$log_name/checkpoints/epoch=000179.ckpt"
echo $ckpt
CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=$PORT \
    pipeline/evaluation/ddp_wrapper.py \
    --module 'timestep_inference_lora' \
    --seed 1000 \
    --lora $LORA \
    --ckpt_path $ckpt \
    --base $config \
    --savedir $res_dir \
    --n_samples 1 \
    --bs 1 --height 320 --width 512 \
    --unconditional_guidance_scale 12.0 \
    --ddim_steps 50 \
    --ddim_eta 1.0 \
    --pretrain 'checkpoints/videocrafter2/model.ckpt' \
    --prompt_file $prompt_file \
    --mid_step $MID_STEP \
    --begin_scale $BEGIN_SCALE \
    --mid_scale $MID_SCALE 

CUDA_VISIBLE_DEVICES=$DEVICES python eval/eval_clip_id.py \
    --video_dir "$res_dir/samples/" \
    --text_dir "$res_dir/input/" \
    --id_img_dir "datasets/benchmark_dataset/wearable_sunglasses2"