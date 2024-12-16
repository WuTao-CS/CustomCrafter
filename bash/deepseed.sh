python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/custom_diffusion.yaml \
    -t --gpus 0,1,2,3 \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> sofa" \
    --datapath ./datasets/benchmark_dataset/furniture_sofa2 \
    --reg_datapath "./datasets/real_reg/samples_sofa/images.txt" \
    --reg_caption "./datasets/real_reg/samples_sofa/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'sofa' \
    --freeze_model "crossattn-kv" \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "furniture_sofa2-wrapper-sofa-customdiffusion"

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/DreamVideo.yaml \
    -t --gpus 0,1,2,3 \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> sofa" \
    --datapath ./datasets/benchmark_dataset/furniture_sofa2 \
    --reg_datapath "./datasets/real_reg/samples_sofa/images.txt" \
    --reg_caption "./datasets/real_reg/samples_sofa/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'sofa' \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "furniture_sofa2-lvdm-wrapper-sofa-dreamvideo"

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/custom_train_wrapper2_lora_reg_loracustom_atten_init_onlylora_ffn_basic_video2.yaml \
    -t --gpus 0,1,2,3 \
    --lora True \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> sofa" \
    --datapath ./datasets/benchmark_dataset/furniture_sofa2 \
    --reg_datapath "./datasets/real_reg/samples_sofa/images.txt" \
    --reg_caption "./datasets/real_reg/samples_sofa/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'sofa' \
    --freeze_model "crossattn" \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "furniture_sofa2-lvdm-wrapper-sofa-crossatten-ours-lr-0.00003-videocrafter2"