python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/custom_diffusion.yaml \
    -t --gpus 0,1,2,3 \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> purse" \
    --datapath ./datasets/benchmark_dataset/luggage_purse4 \
    --reg_datapath "./datasets/real_reg/samples_purse/images.txt" \
    --reg_caption "./datasets/real_reg/samples_purse/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'purse' \
    --freeze_model "crossattn-kv" \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "luggage_purse4-wrapper-purse-customdiffusion"

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/DreamVideo.yaml \
    -t --gpus 0,1,2,3 \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> purse" \
    --datapath ./datasets/benchmark_dataset/luggage_purse4 \
    --reg_datapath "./datasets/real_reg/samples_purse/images.txt" \
    --reg_caption "./datasets/real_reg/samples_purse/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'purse' \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "luggage_purse4-lvdm-wrapper-purse-dreamvideo"

# python -m torch.distributed.run \
#     --nproc_per_node=4 --master_port=1234 \
#     custom.py \
#     --base configs/custom_train_wrapper2_lora_reg_loracustom_atten_init_onlylora_ffn_basic_video2.yaml \
#     -t --gpus 0,1,2,3 \
#     --lora True \
#     --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
#     --caption "<new1> purse" \
#     --datapath ./datasets/benchmark_dataset/luggage_purse4 \
#     --reg_datapath "./datasets/real_reg/samples_purse/images.txt" \
#     --reg_caption "./datasets/real_reg/samples_purse/caption.txt" \
#     --modifier_token "<new1>" \
#     --initializer_token 'purse' \
#     --freeze_model "crossattn" \
#     --with_prior_preservation True \
#     --base_learning_rate 0.00003 \
#     --name "luggage_purse4-lvdm-wrapper-purse-crossatten-lr-0.00003-videocrafter2"