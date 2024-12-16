DEVICES=7
PORT=10124
LORA=True
MID_STEP=5
BEGIN_SCALE=0.4
MID_SCALE=0.8

prompt_file="prompts/purse_prompt.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-31T03-14-47_luggage_purse1-lvdm-wrapper-purse-crossatten-lr-0.00003-videocrafter2"
res_dir="new_results/luggage_purse1/$log_name"+"_time_lora_0.4_0.8_5/"
epochs_name=("epoch=000139" "epoch=000159" "epoch=000179" "epoch=000199")
for name in ${epochs_name[@]}
    do
    ckpt="logs/$log_name/checkpoints/$name.ckpt"
    echo $ckpt
    CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch \
        --nproc_per_node=1 --master_port=$PORT \
        pipeline/evaluation/ddp_wrapper.py \
        --module 'timestep_inference_lora' \
        --seed 1000 \
        --lora $LORA \
        --ckpt_path $ckpt \
        --base $config \
        --savedir $res_dir/$name \
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
        --video_dir "$log_dir/$name/samples/" \
        --text_dir "$log_dir/$name/input/" \
        --id_img_dir "datasets/benchmark_dataset/luggage_purse1"
    done


prompt_file="prompts/purse_prompt.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-31T13-07-06_luggage_purse2-lvdm-wrapper-purse-crossatten-lr-0.00003-videocrafter2"
res_dir="new_results/luggage_purse2/$log_name/"
epochs_name=("epoch=000139" "epoch=000159" "epoch=000179" "epoch=000199")
for name in ${epochs_name[@]}
    do
    ckpt="logs/$log_name/checkpoints/$name.ckpt"
    echo $ckpt
    CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch \
        --nproc_per_node=1 --master_port=$PORT \
        pipeline/evaluation/ddp_wrapper.py \
        --module 'timestep_inference_lora' \
        --seed 1000 \
        --lora $LORA \
        --ckpt_path $ckpt \
        --base $config \
        --savedir $res_dir/$name \
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
        --video_dir "$log_dir/$name/samples/" \
        --text_dir "$log_dir/$name/input/" \
        --id_img_dir "datasets/benchmark_dataset/luggage_purse2"
    done

