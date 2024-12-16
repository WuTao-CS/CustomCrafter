MID_STEP=5
BEGIN_SCALE=0.4
MID_SCALE=0.9

prompt_file="prompts/car_prompts.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-11T11-52-13_transport_car1-lvdm-wrapper-car-crossatten-lr-0.00003-videocrafter2"
res_dir="results/transport_car1/$log_name"+"_time_lora_0.4_0.9_5/"

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
prompt_file="prompts/car_prompts.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-11T23-55-38_transport_car4-lvdm-wrapper-car-crossatten-lr-0.00003-videocrafter2"
res_dir="results/transport_car4/$log_name"+"_time_lora_0.4_0.9_5/"

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
prompt_file="prompts/houseplant_prompts.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-15T05-09-05_houseplant1-lvdm-wrapper-houseplant-crossatten-lr-0.00003-videocrafter2"
res_dir="results/houseplant1/$log_name"+"_time_lora_0.4_0.9_5/"
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
prompt_file="prompts/houseplant_prompts.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-15T15-57-41_houseplant2-lvdm-wrapper-houseplant-crossatten-lr-0.00003-videocrafter2"
res_dir="results/houseplant2/$log_name"+"_time_lora_0.4_0.9_5/"
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
prompt_file="prompts/houseplant_prompts.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-16T02-48-21_houseplant3-lvdm-wrapper-houseplant-crossatten-lr-0.00003-videocrafter2"
res_dir="results/houseplant3/$log_name"+"_time_lora_0.4_0.9_5/"
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
prompt_file="prompts/guitar_prompts.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-16T16-59-18_instrument_music1-lvdm-wrapper-guitar-crossatten-lr-0.00003-videocrafter2"
res_dir="results/instrument_music1/$log_name"+"_time_lora_0.4_0.9_5/"
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
prompt_file="prompts/guitar_prompts.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-17T02-40-16_instrument_music2-lvdm-wrapper-guitar-crossatten-lr-0.00003-videocrafter2"
res_dir="results/instrument_music2/$log_name"+"_time_lora_0.4_0.9_5/"
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

prompt_file="prompts/guitar_prompts.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-17T12-35-45_instrument_music3-lvdm-wrapper-guitar-crossatten-lr-0.00003-videocrafter2"
res_dir="results/instrument_music3/$log_name"+"_time_lora_0.4_0.9_5/"
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
prompt_file="prompts/book_prompts.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-16T13-37-40_book1-lvdm-wrapper-book-crossatten-lr-0.00003-videocrafter2"
res_dir="results/book1/$log_name"+"_time_lora_0.4_0.9_5/"
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
prompt_file="prompts/book_prompts.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-17T00-27-15_book2-lvdm-wrapper-book-crossatten-lr-0.00003-videocrafter2"
res_dir="results/book2/$log_name"+"_time_lora_0.4_0.9_5/"
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
prompt_file="prompts/cup_prompts.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-17T11-19-00_cup1-lvdm-wrapper-cup-crossatten-lr-0.00003-videocrafter2"
res_dir="results/cup1/$log_name"+"_time_lora_0.4_0.9_5/"
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
prompt_file="prompts/cup_prompts.txt"
config=configs/custom_train_wrapper2_lora_test_ffn_without_init_basic_video2.yaml
log_name="2024-03-17T22-07-31_cup2-lvdm-wrapper-cup-crossatten-lr-0.00003-videocrafter2"
res_dir="results/cup2/$log_name"+"_time_lora_0.4_0.9_5/"
epochs_name=("epoch=000019" "epoch=000039" "epoch=000059" "epoch=000079" "epoch=000099" "epoch=000119" "epoch=000139" "epoch=000159" "epoch=000179" "epoch=000199")
for name in ${epochs_name[@]}
    do
    ckpt="logs/$log_name/checkpoints/$name.ckpt"
    echo $ckpt
    CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch \
        --nproc_per_node=1 --master_port=$PORT \
        pipeline/evaluation/ddp_wrapper.py \
        --module 'inference' \
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
        --prompt_file $prompt_file
    done