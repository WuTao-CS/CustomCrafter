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
    --caption "<new1> jacket" \
    --datapath ./datasets/benchmark_dataset/wearable_jacket1 \
    --reg_datapath "./datasets/real_reg/samples_jacket/images.txt" \
    --reg_caption "./datasets/real_reg/samples_jacket/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'jacket' \
    --freeze_model "crossattn-kv" \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "wearable_jacket1-wrapper-jacket-customdiffusion"

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/DreamVideo.yaml \
    -t --gpus 0,1,2,3 \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> jacket" \
    --datapath ./datasets/benchmark_dataset/wearable_jacket1 \
    --reg_datapath "./datasets/real_reg/samples_jacket/images.txt" \
    --reg_caption "./datasets/real_reg/samples_jacket/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'jacket' \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "wearable_jacket1-lvdm-wrapper-jacket-dreamvideo"

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/custom_train_wrapper2_lora_reg_loracustom_atten_init_onlylora_ffn_basic_video2.yaml \
    -t --gpus 0,1,2,3 \
    --lora True \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> jacket" \
    --datapath ./datasets/benchmark_dataset/wearable_jacket1 \
    --reg_datapath "./datasets/real_reg/samples_jacket/images.txt" \
    --reg_caption "./datasets/real_reg/samples_jacket/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'jacket' \
    --freeze_model "crossattn" \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "wearable_jacket1-lvdm-wrapper-jacket-crossatten-ours-lr-0.00003-videocrafter2"


python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/custom_diffusion.yaml \
    -t --gpus 0,1,2,3 \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> jacket" \
    --datapath ./datasets/benchmark_dataset/wearable_jacket2 \
    --reg_datapath "./datasets/real_reg/samples_jacket/images.txt" \
    --reg_caption "./datasets/real_reg/samples_jacket/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'jacket' \
    --freeze_model "crossattn-kv" \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "wearable_jacket2-wrapper-jacket-customdiffusion"

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/DreamVideo.yaml \
    -t --gpus 0,1,2,3 \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> jacket" \
    --datapath ./datasets/benchmark_dataset/wearable_jacket2 \
    --reg_datapath "./datasets/real_reg/samples_jacket/images.txt" \
    --reg_caption "./datasets/real_reg/samples_jacket/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'jacket' \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "wearable_jacket2-lvdm-wrapper-jacket-dreamvideo"

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/custom_train_wrapper2_lora_reg_loracustom_atten_init_onlylora_ffn_basic_video2.yaml \
    -t --gpus 0,1,2,3 \
    --lora True \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> jacket" \
    --datapath ./datasets/benchmark_dataset/wearable_jacket2 \
    --reg_datapath "./datasets/real_reg/samples_jacket/images.txt" \
    --reg_caption "./datasets/real_reg/samples_jacket/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'jacket' \
    --freeze_model "crossattn" \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "wearable_jacket2-lvdm-wrapper-jacket-crossatten-ours-lr-0.00003-videocrafter2"




python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/custom_diffusion.yaml \
    -t --gpus 0,1,2,3 \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> headphone" \
    --datapath ./datasets/benchmark_dataset/things_headphone1 \
    --reg_datapath "./datasets/real_reg/samples_headphone/images.txt" \
    --reg_caption "./datasets/real_reg/samples_headphone/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'headphone' \
    --freeze_model "crossattn-kv" \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "things_headphone1-wrapper-headphone-customdiffusion"

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/DreamVideo.yaml \
    -t --gpus 0,1,2,3 \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> headphone" \
    --datapath ./datasets/benchmark_dataset/things_headphone1 \
    --reg_datapath "./datasets/real_reg/samples_headphone/images.txt" \
    --reg_caption "./datasets/real_reg/samples_headphone/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'headphone' \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "things_headphone1-lvdm-wrapper-headphone-dreamvideo"

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/custom_train_wrapper2_lora_reg_loracustom_atten_init_onlylora_ffn_basic_video2.yaml \
    -t --gpus 0,1,2,3 \
    --lora True \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> headphone" \
    --datapath ./datasets/benchmark_dataset/things_headphone1 \
    --reg_datapath "./datasets/real_reg/samples_headphone/images.txt" \
    --reg_caption "./datasets/real_reg/samples_headphone/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'headphone' \
    --freeze_model "crossattn" \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "things_headphone1-lvdm-wrapper-headphone-crossatten-ours-lr-0.00003-videocrafter2"

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/custom_diffusion.yaml \
    -t --gpus 0,1,2,3 \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> headphone" \
    --datapath ./datasets/benchmark_dataset/things_headphone2 \
    --reg_datapath "./datasets/real_reg/samples_headphone/images.txt" \
    --reg_caption "./datasets/real_reg/samples_headphone/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'headphone' \
    --freeze_model "crossattn-kv" \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "things_headphone2-wrapper-headphone-customdiffusion"

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/DreamVideo.yaml \
    -t --gpus 0,1,2,3 \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> headphone" \
    --datapath ./datasets/benchmark_dataset/things_headphone2 \
    --reg_datapath "./datasets/real_reg/samples_headphone/images.txt" \
    --reg_caption "./datasets/real_reg/samples_headphone/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'headphone' \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "things_headphone2-lvdm-wrapper-headphone-dreamvideo"

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/custom_train_wrapper2_lora_reg_loracustom_atten_init_onlylora_ffn_basic_video2.yaml \
    -t --gpus 0,1,2,3 \
    --lora True \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> headphone" \
    --datapath ./datasets/benchmark_dataset/things_headphone2 \
    --reg_datapath "./datasets/real_reg/samples_headphone/images.txt" \
    --reg_caption "./datasets/real_reg/samples_headphone/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'headphone' \
    --freeze_model "crossattn" \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "things_headphone2-lvdm-wrapper-headphone-crossatten-ours-lr-0.00003-videocrafter2"

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/custom_diffusion.yaml \
    -t --gpus 0,1,2,3 \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> shoes" \
    --datapath ./datasets/benchmark_dataset/wearable_shoes1 \
    --reg_datapath "./datasets/real_reg/samples_shoes/images.txt" \
    --reg_caption "./datasets/real_reg/samples_shoes/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'shoes' \
    --freeze_model "crossattn-kv" \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "wearable_shoes1-wrapper-shoes-customdiffusion"

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/DreamVideo.yaml \
    -t --gpus 0,1,2,3 \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> shoes" \
    --datapath ./datasets/benchmark_dataset/wearable_shoes1 \
    --reg_datapath "./datasets/real_reg/samples_shoes/images.txt" \
    --reg_caption "./datasets/real_reg/samples_shoes/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'shoes' \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "wearable_shoes1-lvdm-wrapper-shoes-dreamvideo"

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/custom_train_wrapper2_lora_reg_loracustom_atten_init_onlylora_ffn_basic_video2.yaml \
    -t --gpus 0,1,2,3 \
    --lora True \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> shoes" \
    --datapath ./datasets/benchmark_dataset/wearable_shoes1 \
    --reg_datapath "./datasets/real_reg/samples_shoes/images.txt" \
    --reg_caption "./datasets/real_reg/samples_shoes/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'shoes' \
    --freeze_model "crossattn" \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "wearable_shoes1-lvdm-wrapper-shoes-crossatten-ours-lr-0.00003-videocrafter2"

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/custom_diffusion.yaml \
    -t --gpus 0,1,2,3 \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> shoes" \
    --datapath ./datasets/benchmark_dataset/wearable_shoes2 \
    --reg_datapath "./datasets/real_reg/samples_shoes/images.txt" \
    --reg_caption "./datasets/real_reg/samples_shoes/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'shoes' \
    --freeze_model "crossattn-kv" \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "wearable_shoes2-wrapper-shoes-customdiffusion"

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/DreamVideo.yaml \
    -t --gpus 0,1,2,3 \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> shoes" \
    --datapath ./datasets/benchmark_dataset/wearable_shoes2 \
    --reg_datapath "./datasets/real_reg/samples_shoes/images.txt" \
    --reg_caption "./datasets/real_reg/samples_shoes/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'shoes' \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "wearable_shoes2-lvdm-wrapper-shoes-dreamvideo"

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/custom_train_wrapper2_lora_reg_loracustom_atten_init_onlylora_ffn_basic_video2.yaml \
    -t --gpus 0,1,2,3 \
    --lora True \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> shoes" \
    --datapath ./datasets/benchmark_dataset/wearable_shoes2 \
    --reg_datapath "./datasets/real_reg/samples_shoes/images.txt" \
    --reg_caption "./datasets/real_reg/samples_shoes/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'shoes' \
    --freeze_model "crossattn" \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "wearable_shoes2-lvdm-wrapper-shoes-crossatten-ours-lr-0.00003-videocrafter2"

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/custom_diffusion.yaml \
    -t --gpus 0,1,2,3 \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> sofa" \
    --datapath ./datasets/benchmark_dataset/furniture_sofa1 \
    --reg_datapath "./datasets/real_reg/samples_sofa/images.txt" \
    --reg_caption "./datasets/real_reg/samples_sofa/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'sofa' \
    --freeze_model "crossattn-kv" \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "furniture_sofa1-wrapper-sofa-customdiffusion"

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/DreamVideo.yaml \
    -t --gpus 0,1,2,3 \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> sofa" \
    --datapath ./datasets/benchmark_dataset/furniture_sofa1 \
    --reg_datapath "./datasets/real_reg/samples_sofa/images.txt" \
    --reg_caption "./datasets/real_reg/samples_sofa/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'sofa' \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "furniture_sofa1-lvdm-wrapper-sofa-dreamvideo"

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    custom.py \
    --base configs/custom_train_wrapper2_lora_reg_loracustom_atten_init_onlylora_ffn_basic_video2.yaml \
    -t --gpus 0,1,2,3 \
    --lora True \
    --resume-from-checkpoint-custom 'checkpoints/videocrafter2/model.ckpt' \
    --caption "<new1> sofa" \
    --datapath ./datasets/benchmark_dataset/furniture_sofa1 \
    --reg_datapath "./datasets/real_reg/samples_sofa/images.txt" \
    --reg_caption "./datasets/real_reg/samples_sofa/caption.txt" \
    --modifier_token "<new1>" \
    --initializer_token 'sofa' \
    --freeze_model "crossattn" \
    --with_prior_preservation True \
    --base_learning_rate 0.00003 \
    --name "furniture_sofa1-lvdm-wrapper-sofa-crossatten-ours-lr-0.00003-videocrafter2"

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