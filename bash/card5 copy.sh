MID_STEP=5
BEGIN_SCALE=0.4
MID_SCALE=0.9

prompt_file="prompts/cup_prompts.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-18T09-23-38_cup3-lvdm-wrapper-cup-crossatten-lr-0.00003-videocrafter2"
res_dir="results/cup3/$log_name"+"_time_lora_0.4_0.9_5/"
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
    done


prompt_file="prompts/sunglasses_prompts.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-17T22-47-30_sunglasses1-lvdm-wrapper-sunglasses-crossatten-lr-0.00003-videocrafter2"
res_dir="results/sunglasses1/$log_name"+"_time_lora_0.4_0.9_5/"
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
    done

prompt_file="prompts/sunglasses_prompts.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-18T08-53-25_sunglasses2-lvdm-wrapper-sunglasses-crossatten-lr-0.00003-videocrafter2"
res_dir="results/sunglasses2/$log_name"+"_time_lora_0.4_0.9_5/"
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
    done

prompt_file="prompts/jacket_prompts.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-18T04-51-54_wearable_jacket1-lvdm-wrapper-jacket-crossatten-ours-lr-0.00003-videocrafter2"
res_dir="results/wearable_jacket1/$log_name"+"_time_lora_0.4_0.9_5/"
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
    done

prompt_file="prompts/jacket_prompts.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-18T16-03-03_wearable_jacket2-lvdm-wrapper-jacket-crossatten-ours-lr-0.00003-videocrafter2"
res_dir="results/wearable_jacket2/$log_name"+"_time_lora_0.4_0.9_5/"
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
    done

prompt_file="prompts/headphone_prompts.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-19T02-51-30_things_headphone1-lvdm-wrapper-headphone-crossatten-ours-lr-0.00003-videocrafter2"
res_dir="results/things_headphone1/$log_name"+"_time_lora_0.4_0.9_5/"
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
    done

prompt_file="prompts/headphone_prompts.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-19T13-41-08_things_headphone2-lvdm-wrapper-headphone-crossatten-ours-lr-0.00003-videocrafter2"
res_dir="results/things_headphone2/$log_name"+"_time_lora_0.4_0.9_5/"
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
    done


prompt_file="prompts/shoes_prompts.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-20T00-32-31_wearable_shoes1-lvdm-wrapper-shoes-crossatten-ours-lr-0.00003-videocrafter2"
res_dir="results/wearable_shoes1/$log_name"+"_time_lora_0.4_0.9_5/"
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
    done

prompt_file="prompts/shoes_prompts.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-20T11-20-54_wearable_shoes2-lvdm-wrapper-shoes-crossatten-ours-lr-0.00003-videocrafter2"
res_dir="results/wearable_shoes2/$log_name"+"_time_lora_0.4_0.9_5/"
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
    done

prompt_file="prompts/sofa_prompts.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-20T22-13-26_furniture_sofa1-lvdm-wrapper-sofa-crossatten-ours-lr-0.00003-videocrafter2"
res_dir="results/furniture_sofa1/$log_name"+"_time_lora_0.4_0.9_5/"
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
    done

prompt_file="prompts/sofa_prompts.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-21T09-06-39_furniture_sofa2-lvdm-wrapper-sofa-crossatten-ours-lr-0.00003-videocrafter2"
res_dir="results/furniture_sofa1/$log_name"+"_time_lora_0.4_0.9_5/"
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
    done