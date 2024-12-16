# pip install clip_retrieval

CUDA_VISIBLE_DEVICES=0 nohup python sample_reg.py --prompt "video of a <new1> rabbit plush toy" --outdir ./datasets/real_reg/samples_rabbit_plush_toy > sample1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python sample_reg.py --prompt "video of a <new1> cow plush toy" --outdir ./datasets/real_reg/samples_cow_plush_toy > sample2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python sample_reg.py --prompt "video of a <new1> dice plush toy" --outdir ./datasets/real_reg/samples_dice_plush_toy > sample3.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python sample_reg.py --prompt "video of a <new1> unicorn plush toy" --outdir ./datasets/real_reg/samples_unicorn_plush_toy > sample4.log 2>&1 &
