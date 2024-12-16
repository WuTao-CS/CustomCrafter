nvidia-smi
conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai 
cd /group/40034/jerryxwli/code/VideoCrafter_Share/


python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom_vision.py \
    --base configs/custom_train_wrapper2_lora_reg_loracustom_atten_init_onlylora_ffn_basic_video2_visual_v100.yaml \
    -t --gpus 0,1,2,3 \
    --lora True \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> person" \
    --datapath ./datasets/characters/anime/Hina_Amano/raw_data \
    --reg_datapath "./datasets/real_reg/samples_person/images.txt" \
    --reg_caption "./datasets/real_reg/samples_person/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'person' \
    --freeze_model "crossattn" \
    --base_learning_rate 0.00003 \
    --name "Hina_Amano-wrapper-person-ours-v100"

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=13234 \
    custom_vision.py \
    --base configs/custom_train_wrapper2_lora_reg_loracustom_atten_init_onlylora_ffn_basic_video2_visual_v100.yaml \
    -t --gpus 4,5,6,7 \
    --lora True \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> person" \
    --datapath ./datasets/characters/anime/Tezuka_Kunimitsu/raw_data \
    --reg_datapath "./datasets/real_reg/samples_person/images.txt" \
    --reg_caption "./datasets/real_reg/samples_person/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'person' \
    --freeze_model "crossattn" \
    --base_learning_rate 0.00003 \
    --name "Tezuka_Kunimitsu-wrapper-person-ours-v100"