nvidia-smi
conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai 
cd /group/40034/jerryxwli/code/VideoCrafter_Share/

python -m torch.distributed.run \
    --nproc_per_node=1 --master_port=1234 \
    main.py \
    --base configs/textual_inversion.yaml \
    -t \
    --actual_resume /group/40034/jerryxwli/code/VideoCrafter_Share/checkpoints/model.ckpt \
    -n teddybear \
    --gpus 0,1,2,3 \
    --init_word teddybear \
    --placeholder_string "<new1>" \
    lightning.trainer.num_nodes=1