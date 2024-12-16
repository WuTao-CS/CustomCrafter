DEVICES=0
PORT=10234

prompt_file="prompts/car_prompts.txt"
config=configs/custom_diffusion_test.yaml
log_name="2024-03-11T03-38-13_transport_car1-wrapper-cat-customdiffusion"
res_dir="new_results/transport_car1/$log_name/"
epochs_name=("epoch=000019" "epoch=000079" "epoch=000099" "epoch=000179" "epoch=000199")
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
    done


prompt_file="prompts/car_prompts.txt"
config=configs/custom_diffusion_test.yaml
log_name="2024-03-11T15-36-35_transport_car4-wrapper-cat-customdiffusion"
res_dir="new_results/transport_car4/$log_name/"
epochs_name=("epoch=000019" "epoch=000079" "epoch=000099" "epoch=000179" "epoch=000199")
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
    done

prompt_file="prompts/houseplant_prompts.txt"
config=configs/custom_diffusion_test.yaml
log_name="2024-03-13T22-51-27_houseplant1-wrapper-houseplant-customdiffusion"
res_dir="new_results/houseplant1/$log_name/"
epochs_name=("epoch=000019" "epoch=000079" "epoch=000099" "epoch=000179" "epoch=000199")
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
    done

prompt_file="prompts/houseplant_prompts.txt"
config=configs/custom_diffusion_test.yaml
log_name="2024-03-15T08-23-15_houseplant2-wrapper-houseplant-customdiffusion"
res_dir="new_results/houseplant2/$log_name/"
epochs_name=("epoch=000019" "epoch=000079" "epoch=000099" "epoch=000179" "epoch=000199")
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
    done

prompt_file="prompts/houseplant_prompts.txt"
config=configs/custom_diffusion_test.yaml
log_name="2024-03-15T19-11-19_houseplant3-wrapper-houseplant-customdiffusion"
res_dir="new_results/houseplant3/$log_name/"
epochs_name=("epoch=000019" "epoch=000079" "epoch=000099" "epoch=000179" "epoch=000199")
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
    done

prompt_file="prompts/guitar_prompts.txt"
config=configs/custom_diffusion_test.yaml
log_name="2024-03-14T21-07-49_instrument_music1-wrapper-guitar-customdiffusion"
res_dir="new_results/instrument_music1/$log_name/"
epochs_name=("epoch=000019" "epoch=000079" "epoch=000099" "epoch=000179" "epoch=000199")
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
    done

prompt_file="prompts/guitar_prompts.txt"
config=configs/custom_diffusion_test.yaml
log_name="2024-03-16T20-09-38_instrument_music2-wrapper-guitar-customdiffusion"
res_dir="new_results/instrument_music2/$log_name/"
epochs_name=("epoch=000019" "epoch=000079" "epoch=000099" "epoch=000179" "epoch=000199")
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
    done

prompt_file="prompts/guitar_prompts.txt"
config=configs/custom_diffusion_test.yaml
log_name="2024-03-17T05-50-21_instrument_music3-wrapper-guitar-customdiffusion"
res_dir="new_results/instrument_music3/$log_name/"
epochs_name=("epoch=000019" "epoch=000079" "epoch=000099" "epoch=000179" "epoch=000199")
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
    done

prompt_file="prompts/book_prompts.txt"
config=configs/custom_diffusion_test.yaml
log_name="2024-03-16T06-02-59_book1-wrapper-book-customdiffusion"
res_dir="new_results/book1/$log_name/"
epochs_name=("epoch=000019" "epoch=000079" "epoch=000099" "epoch=000179" "epoch=000199")
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
    done

prompt_file="prompts/book_prompts.txt"
config=configs/custom_diffusion_test.yaml
log_name="2024-03-16T16-51-32_book2-wrapper-book-customdiffusion"
res_dir="new_results/book2/$log_name/"
epochs_name=("epoch=000019" "epoch=000079" "epoch=000099" "epoch=000179" "epoch=000199")
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
    done

prompt_file="prompts/cup_prompts.txt"
config=configs/custom_diffusion_test.yaml
log_name="2024-03-17T03-41-09_cup1-wrapper-cup-customdiffusion"
res_dir="new_results/cup1/$log_name/"
epochs_name=("epoch=000019" "epoch=000079" "epoch=000099" "epoch=000179" "epoch=000199")
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
    done

prompt_file="prompts/cup_prompts.txt"
config=configs/custom_diffusion_test.yaml
log_name="2024-03-17T14-32-37_cup2-wrapper-cup-customdiffusion"
res_dir="new_results/cup2/$log_name/"
epochs_name=("epoch=000019" "epoch=000079" "epoch=000099" "epoch=000179" "epoch=000199")
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
    done