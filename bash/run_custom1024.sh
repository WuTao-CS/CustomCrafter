DEVICES=0
PORT=10234
LORA=True
prompt_file="prompts/teddybear_move_prompts.txt"
PORT=17123
DEVICES=1
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic.yaml
log_name="2024-01-18T19-26-48_plushie-teddybear-lvdm-wrapper-teddybear-plush-toy-crossatten-without-init-onlylora-ffn-basic-lr-0.00003"
res_dir="results/mid_lora_10/move_$log_name/"
name="epoch=000179"
ckpt="logs/$log_name/checkpoints/$name.ckpt"

CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=$PORT \
    pipeline/evaluation/ddp_wrapper.py \
    --module 'timestep_inference_lora' \
    --seed 1000 \
    --lora $LORA \
    --ckpt_path $ckpt \
    --base $config \
    --savedir $res_dir/$name \
    --mid_step 10 \
    --n_samples 1 \
    --bs 1 --height 320 --width 512 \
    --unconditional_guidance_scale 15.0 \
    --ddim_steps 50 \
    --ddim_eta 1.0 \
    --prompt_file $prompt_file