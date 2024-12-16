python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/custom_diffusion.yaml \
    -t --gpus 0,1,2,3 \
    --resume-from-checkpoint-custom  checkpoints/videocrafter2/model.ckpt \
    --caption "<new1> teddybear" \
    --datapath ./datasets/benchmark_dataset/plushie_teddybear \
    --reg_datapath "./datasets/real_reg/samples_teddybear/images.txt" \
    --reg_caption "./datasets/real_reg/samples_teddybear/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token teddybear \
    --freeze_model "updateall" \
    --name "plushie-teddybear-lvdm-wrapper-teddybear-updateall"