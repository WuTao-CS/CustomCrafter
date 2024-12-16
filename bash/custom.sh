nvidia-smi
conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai 
cd /group/40034/jerryxwli/code/VideoCrafter_Share/
# python sample_reg.py \
#     --prompt 'video of a anime girl' \
#     --outdir 'datasets/real_reg/sample_anime_girl/'

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/train_customcrafter.yaml \
    -t --gpus 0,1,2,3 \
    --lora True \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> cat" \
    --modifier_token "<new1>" \
    --initializer_token 'cat' \
    --datapath "datasets/benchmark_dataset/pet_cat5/" \
    --reg_datapath "./datasets/real_reg/samples_cat/images.txt" \
    --reg_caption "./datasets/real_reg/samples_cat/caption.txt" \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "Miyazono_CustomCrafter_Videocrafter2_Checkpoints"
