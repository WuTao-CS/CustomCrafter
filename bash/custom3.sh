nvidia-smi
conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai 
cd /group/40034/jerryxwli/code/VideoCrafter_Share/

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/custom_diffusion.yaml \
    -t --gpus 0,1,2,3 \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> dog" \
    --datapath ./datasets/benchmark_dataset/pet_dog1 \
    --reg_datapath "./datasets/real_reg/samples_dog/images.txt" \
    --reg_caption "./datasets/real_reg/samples_dog/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'dog' \
    --freeze_model "crossattn-kv" \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "pet_dog1-wrapper-dog-customdiffusion"

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/DreamVideo.yaml \
    -t --gpus 0,1,2,3 \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> dog" \
    --datapath ./datasets/benchmark_dataset/pet_dog1 \
    --reg_datapath "./datasets/real_reg/samples_dog/images.txt" \
    --reg_caption "./datasets/real_reg/samples_dog/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'dog' \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "pet_dog1-lvdm-wrapper-dog-dreamvideo"

# python -m torch.distributed.run \
#     --nproc_per_node=4 --master_port=1234 \
#     custom.py \
#     --base configs/custom_train_wrapper2_lora_reg_loracustom_atten_init_onlylora_ffn_basic_video2.yaml \
#     -t --gpus 0,1,2,3 \
#     --lora True \
#     --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
#     --caption "<new1> dog" \
#     --datapath ./datasets/benchmark_dataset/pet_dog1 \
#     --reg_datapath "./datasets/real_reg/samples_dog/images.txt" \
#     --reg_caption "./datasets/real_reg/samples_dog/caption.txt" \
#     --modifier_token "<new1>" \
#     --initializer_token 'dog' \
#     --freeze_model "crossattn" \
#     --with_prior_preservation True \
#     --base_learning_rate 0.00003 \
#     --name "plushie_bunny-lvdm-wrapper-dog-crossatten-lr-0.00003-videocrafter2"

# python -m torch.distributed.run \
#     --nproc_per_node=4 --master_port=1234 \
#     custom.py \
#     --base configs/custom_diffusion.yaml \
#     -t --gpus 0,1,2,3 \
#     --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
#     --caption "<new1> cow plush toy" \
#     --datapath ./datasets/benchmark_dataset/plushie_cow \
#     --reg_datapath "./datasets/real_reg/samples_cow_plush_toy/images.txt" \
#     --reg_caption "./datasets/real_reg/samples_cow_plush_toy/caption.txt" \
#     --modifier_token "<new1>" \
#     --initializer_token 'cow plush toy' \
#     --freeze_model "crossattn-kv" \
#     --with_prior_preservation True \
#     --base_learning_rate 0.00003 \
#     --name "plushie_cow-wrapper-cow_plush_toy-customdiffusion"

# python -m torch.distributed.run \
#     --nproc_per_node=4 --master_port=1234 \
#     custom.py \
#     --base configs/DreamVideo.yaml \
#     -t --gpus 0,1,2,3 \
#     --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
#     --caption "<new1> cow plush toy" \
#     --datapath ./datasets/benchmark_dataset/plushie_cow \
#     --reg_datapath "./datasets/real_reg/samples_cow_plush_toy/images.txt" \
#     --reg_caption "./datasets/real_reg/samples_cow_plush_toy/caption.txt" \
#     --modifier_token "<new1>" \
#     --initializer_token 'cow plush toy' \
#     --with_prior_preservation True \
#     --base_learning_rate 0.00003 \
#     --name "plushie_cow-lvdm-wrapper-cow_plush_toy-dreamvideo"

# python -m torch.distributed.run \
#     --nproc_per_node=4 --master_port=1234 \
#     custom.py \
#     --base configs/custom_train_wrapper2_lora_reg_loracustom_atten_init_onlylora_ffn_basic_video2.yaml \
#     -t --gpus 0,1,2,3 \
#     --lora True \
#     --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
#     --caption "<new1> cow plush toy" \
#     --datapath ./datasets/benchmark_dataset/plushie_cow \
#     --reg_datapath "./datasets/real_reg/samples_cow_plush_toy/images.txt" \
#     --reg_caption "./datasets/real_reg/samples_cow_plush_toy/caption.txt" \
#     --modifier_token "<new1>" \
#     --initializer_token 'cow plush toy' \
#     --freeze_model "crossattn" \
#     --with_prior_preservation True \
#     --base_learning_rate 0.00003 \
#     --name "plushie_cow-lvdm-wrapper-cow_plush_toy-crossatten-lr-0.00003-videocrafter2"

# python -m torch.distributed.run \
#     --nproc_per_node=4 --master_port=1234 \
#     custom.py \
#     --base configs/custom_diffusion.yaml \
#     -t --gpus 0,1,2,3 \
#     --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
#     --caption "<new1> dice plush toy" \
#     --datapath ./datasets/benchmark_dataset/plushie_dice \
#     --reg_datapath "./datasets/real_reg/samples_dice_plush_toy/images.txt" \
#     --reg_caption "./datasets/real_reg/samples_dice_plush_toy/caption.txt" \
#     --modifier_token "<new1>" \
#     --initializer_token 'dice plush toy' \
#     --freeze_model "crossattn-kv" \
#     --with_prior_preservation True \
#     --base_learning_rate 0.00003 \
#     --name "plushie_dice-wrapper-dice_plush_toy-customdiffusion"

# python -m torch.distributed.run \
#     --nproc_per_node=4 --master_port=1234 \
#     custom.py \
#     --base configs/DreamVideo.yaml \
#     -t --gpus 0,1,2,3 \
#     --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
#     --caption "<new1> dice plush toy" \
#     --datapath ./datasets/benchmark_dataset/plushie_dice \
#     --reg_datapath "./datasets/real_reg/samples_dice_plush_toy/images.txt" \
#     --reg_caption "./datasets/real_reg/samples_dice_plush_toy/caption.txt" \
#     --modifier_token "<new1>" \
#     --initializer_token 'dice plush toy' \
#     --with_prior_preservation True \
#     --base_learning_rate 0.00003 \
#     --name "plushie_dice-lvdm-wrapper-dice_plush_toy-dreamvideo"

# python -m torch.distributed.run \
#     --nproc_per_node=4 --master_port=1234 \
#     custom.py \
#     --base configs/custom_train_wrapper2_lora_reg_loracustom_atten_init_onlylora_ffn_basic_video2.yaml \
#     -t --gpus 0,1,2,3 \
#     --lora True \
#     --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
#     --caption "<new1> dice plush toy" \
#     --datapath ./datasets/benchmark_dataset/plushie_dice \
#     --reg_datapath "./datasets/real_reg/samples_dice_plush_toy/images.txt" \
#     --reg_caption "./datasets/real_reg/samples_dice_plush_toy/caption.txt" \
#     --modifier_token "<new1>" \
#     --initializer_token 'dice plush toy' \
#     --freeze_model "crossattn" \
#     --with_prior_preservation True \
#     --base_learning_rate 0.00003 \
#     --name "plushie_dice-lvdm-wrapper-dice_plush_toy-crossatten-lr-0.00003-videocrafter2"

# python -m torch.distributed.run \
#     --nproc_per_node=4 --master_port=1234 \
#     custom.py \
#     --base configs/custom_diffusion.yaml \
#     -t --gpus 0,1,2,3 \
#     --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
#     --caption "<new1> unicorn plush toy" \
#     --datapath ./datasets/benchmark_dataset/plushie_unicorn \
#     --reg_datapath "./datasets/real_reg/samples_unicorn_plush_toy/images.txt" \
#     --reg_caption "./datasets/real_reg/samples_unicorn_plush_toy/caption.txt" \
#     --modifier_token "<new1>" \
#     --initializer_token 'unicorn plush toy' \
#     --freeze_model "crossattn-kv" \
#     --with_prior_preservation True \
#     --base_learning_rate 0.00003 \
#     --name "plushie_unicorn-wrapper-unicorn_plush_toy-customdiffusion"

# python -m torch.distributed.run \
#     --nproc_per_node=4 --master_port=1234 \
#     custom.py \
#     --base configs/DreamVideo.yaml \
#     -t --gpus 0,1,2,3 \
#     --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
#     --caption "<new1> unicorn plush toy" \
#     --datapath ./datasets/benchmark_dataset/plushie_unicorn \
#     --reg_datapath "./datasets/real_reg/samples_unicorn_plush_toy/images.txt" \
#     --reg_caption "./datasets/real_reg/samples_unicorn_plush_toy/caption.txt" \
#     --modifier_token "<new1>" \
#     --initializer_token 'unicorn plush toy' \
#     --with_prior_preservation True \
#     --base_learning_rate 0.00003 \
#     --name "plushie_unicorn-lvdm-wrapper-unicorn_plush_toy-dreamvideo"

# python -m torch.distributed.run \
#     --nproc_per_node=4 --master_port=1234 \
#     custom.py \
#     --base configs/custom_train_wrapper2_lora_reg_loracustom_atten_init_onlylora_ffn_basic_video2.yaml \
#     -t --gpus 0,1,2,3 \
#     --lora True \
#     --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
#     --caption "<new1> unicorn plush toy" \
#     --datapath ./datasets/benchmark_dataset/plushie_unicorn \
#     --reg_datapath "./datasets/real_reg/samples_unicorn_plush_toy/images.txt" \
#     --reg_caption "./datasets/real_reg/samples_unicorn_plush_toy/caption.txt" \
#     --modifier_token "<new1>" \
#     --initializer_token 'unicorn plush toy' \
#     --freeze_model "crossattn" \
#     --with_prior_preservation True \
#     --base_learning_rate 0.00003 \
#     --name "plushie_unicorn-lvdm-wrapper-unicorn_plush_toy-crossatten-lr-0.00003-videocrafter2"