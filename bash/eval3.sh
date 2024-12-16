echo "###############################"
log_dir="new_results/wearable_jacket1/2024-03-17T20-51-10_wearable_jacket1-wrapper-jacket-customdiffusion"
epochs_name=("epoch=000019" "epoch=000079" "epoch=000099" "epoch=000179" "epoch=000199")
for name in ${epochs_name[@]}
    do
    python eval/eval_clip_id.py \
        --video_dir "$log_dir/$name/samples/" \
        --text_dir "$log_dir/$name/input/" \
        --id_img_dir "datasets/benchmark_dataset/wearable_jacket1"
    done

log_dir="new_results/wearable_jacket1/2024-03-18T00-24-44_wearable_jacket1-lvdm-wrapper-jacket-dreamvideo"
epochs_name=("epoch=000039" "epoch=000079" "epoch=000099" "epoch=000179" "epoch=000199")
for name in ${epochs_name[@]}
    do
    python eval/eval_clip_id.py \
        --video_dir "$log_dir/$name/samples/" \
        --text_dir "$log_dir/$name/input/" \
        --id_img_dir "datasets/benchmark_dataset/wearable_jacket1"
    done

log_dir="new_results/wearable_jacket1/2024-03-18T04-51-54_wearable_jacket1-lvdm-wrapper-jacket-crossatten-ours-lr-0.00003-videocrafter2"
epochs_name=("epoch=000139" "epoch=000159" "epoch=000179" "epoch=000199")
for name in ${epochs_name[@]}
    do
    python eval/eval_clip_id.py \
        --video_dir "$log_dir/$name/samples/" \
        --text_dir "$log_dir/$name/input/" \
        --id_img_dir "datasets/benchmark_dataset/wearable_jacket1"
    done

log_dir="new_results/wearable_jacket1/2024-03-18T04-51-54_wearable_jacket1-lvdm-wrapper-jacket-crossatten-ours-lr-0.00003-videocrafter2+_time_lora_0.4_0.8_5"
epochs_name=("epoch=000139" "epoch=000159" "epoch=000179" "epoch=000199")
for name in ${epochs_name[@]}
    do
    python eval/eval_clip_id.py \
        --video_dir "$log_dir/$name/samples/" \
        --text_dir "$log_dir/$name/input/" \
        --id_img_dir "datasets/benchmark_dataset/wearable_jacket1"
    done

echo "###############################"
log_dir="new_results/wearable_jacket2/2024-03-18T08-06-05_wearable_jacket2-wrapper-jacket-customdiffusion"
epochs_name=("epoch=000019" "epoch=000079" "epoch=000099" "epoch=000179" "epoch=000199")
for name in ${epochs_name[@]}
    do
    python eval/eval_clip_id.py \
        --video_dir "$log_dir/$name/samples/" \
        --text_dir "$log_dir/$name/input/" \
        --id_img_dir "datasets/benchmark_dataset/wearable_jacket2"
    done

log_dir="new_results/wearable_jacket2/2024-03-18T11-13-58_wearable_jacket2-lvdm-wrapper-jacket-dreamvideo"
epochs_name=("epoch=000039" "epoch=000079" "epoch=000099" "epoch=000179" "epoch=000199")
for name in ${epochs_name[@]}
    do
    python eval/eval_clip_id.py \
        --video_dir "$log_dir/$name/samples/" \
        --text_dir "$log_dir/$name/input/" \
        --id_img_dir "datasets/benchmark_dataset/wearable_jacket2"
    done

log_dir="new_results/wearable_jacket2/2024-03-18T16-03-03_wearable_jacket2-lvdm-wrapper-jacket-crossatten-ours-lr-0.00003-videocrafter2"
epochs_name=("epoch=000139" "epoch=000159" "epoch=000179" "epoch=000199")
for name in ${epochs_name[@]}
    do
    python eval/eval_clip_id.py \
        --video_dir "$log_dir/$name/samples/" \
        --text_dir "$log_dir/$name/input/" \
        --id_img_dir "datasets/benchmark_dataset/wearable_jacket2"
    done

log_dir="new_results/wearable_jacket2/2024-03-18T16-03-03_wearable_jacket2-lvdm-wrapper-jacket-crossatten-ours-lr-0.00003-videocrafter2+_time_lora_0.4_0.8_5"
epochs_name=("epoch=000139" "epoch=000159" "epoch=000179" "epoch=000199")
for name in ${epochs_name[@]}
    do
    python eval/eval_clip_id.py \
        --video_dir "$log_dir/$name/samples/" \
        --text_dir "$log_dir/$name/input/" \
        --id_img_dir "datasets/benchmark_dataset/wearable_jacket2"
    done