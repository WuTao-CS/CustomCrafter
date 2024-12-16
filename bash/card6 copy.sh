DEVICES=5
PORT=9193

prompt_file="prompts/purse_prompt.txt"
config=configs/DreamVideo.yaml
log_name="2024-03-30T23-34-59_luggage_purse1-lvdm-wrapper-purse-dreamvideo"
res_dir="new_results/luggage_purse1/$log_name/"
epochs_name=("epoch=000039" "epoch=000079" "epoch=000099" "epoch=000179" "epoch=000199")
for name in ${epochs_name[@]}
    do
    ckpt="logs/$log_name/checkpoints/$name.ckpt"
    echo $ckpt
    CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch \
        --nproc_per_node=1 --master_port=$PORT \
        pipeline/evaluation/ddp_wrapper.py \
        --module 'inference' \
        --seed 1000 \
        --ckpt_path $ckpt \
        --base $config \
        --savedir $res_dir/$name \
        --n_samples 1 \
        --bs 1 --height 320 --width 512 \
        --unconditional_guidance_scale 12.0 \
        --ddim_steps 50 \
        --ddim_eta 1.0 \
        --pretrain 'checkpoints/videocrafter2/model.ckpt' \
        --prompt_file $prompt_file
    CUDA_VISIBLE_DEVICES=$DEVICES python eval/eval_clip_id.py \
        --video_dir "$log_dir/$name/samples/" \
        --text_dir "$log_dir/$name/input/" \
        --id_img_dir "datasets/benchmark_dataset/luggage_purse1"
    done


prompt_file="prompts/purse_prompt.txt"
config=configs/DreamVideo.yaml
log_name="2024-03-31T09-29-59_luggage_purse2-lvdm-wrapper-purse-dreamvideo"
res_dir="new_results/luggage_purse2/$log_name/"
epochs_name=("epoch=000039" "epoch=000079" "epoch=000099" "epoch=000179" "epoch=000199")
for name in ${epochs_name[@]}
    do
    ckpt="logs/$log_name/checkpoints/$name.ckpt"
    echo $ckpt
    CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch \
        --nproc_per_node=1 --master_port=$PORT \
        pipeline/evaluation/ddp_wrapper.py \
        --module 'inference' \
        --seed 1000 \
        --ckpt_path $ckpt \
        --base $config \
        --savedir $res_dir/$name \
        --n_samples 1 \
        --bs 1 --height 320 --width 512 \
        --unconditional_guidance_scale 12.0 \
        --ddim_steps 50 \
        --ddim_eta 1.0 \
        --pretrain 'checkpoints/videocrafter2/model.ckpt' \
        --prompt_file $prompt_file
    CUDA_VISIBLE_DEVICES=$DEVICES python eval/eval_clip_id.py \
        --video_dir "$log_dir/$name/samples/" \
        --text_dir "$log_dir/$name/input/" \
        --id_img_dir "datasets/benchmark_dataset/luggage_purse2"
    done