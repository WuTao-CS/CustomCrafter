echo "###############################"
log_dir="new_results/wearable_shoes1/2024-03-19T16-54-39_wearable_shoes1-wrapper-shoes-customdiffusion"
epochs_name=("epoch=000019" "epoch=000079" "epoch=000099" "epoch=000179" "epoch=000199")
for name in ${epochs_name[@]}
    do
    python eval/eval_clip_id.py \
        --video_dir "$log_dir/$name/samples/" \
        --text_dir "$log_dir/$name/input/" \
        --id_img_dir "datasets/benchmark_dataset/wearable_shoes1"
    done

log_dir="new_results/wearable_shoes1/2024-03-19T20-02-46_wearable_shoes1-lvdm-wrapper-shoes-dreamvideo"
epochs_name=("epoch=000039" "epoch=000079" "epoch=000099" "epoch=000179" "epoch=000199")
for name in ${epochs_name[@]}
    do
    python eval/eval_clip_id.py \
        --video_dir "$log_dir/$name/samples/" \
        --text_dir "$log_dir/$name/input/" \
        --id_img_dir "datasets/benchmark_dataset/wearable_shoes1"
    done

log_dir="new_results/wearable_shoes1/2024-03-20T00-32-31_wearable_shoes1-lvdm-wrapper-shoes-crossatten-ours-lr-0.00003-videocrafter2"
epochs_name=("epoch=000139" "epoch=000159" "epoch=000179" "epoch=000199")
for name in ${epochs_name[@]}
    do
    python eval/eval_clip_id.py \
        --video_dir "$log_dir/$name/samples/" \
        --text_dir "$log_dir/$name/input/" \
        --id_img_dir "datasets/benchmark_dataset/wearable_shoes1"
    done

log_dir="new_results/wearable_shoes1/2024-03-20T00-32-31_wearable_shoes1-lvdm-wrapper-shoes-crossatten-ours-lr-0.00003-videocrafter2+_time_lora_0.4_0.8_5"
epochs_name=("epoch=000139" "epoch=000159" "epoch=000179" "epoch=000199")
for name in ${epochs_name[@]}
    do
    python eval/eval_clip_id.py \
        --video_dir "$log_dir/$name/samples/" \
        --text_dir "$log_dir/$name/input/" \
        --id_img_dir "datasets/benchmark_dataset/wearable_shoes1"
    done

echo "###############################"
log_dir="new_results/wearable_shoes2/2024-03-20T03-46-38_wearable_shoes2-wrapper-shoes-customdiffusion"
epochs_name=("epoch=000019" "epoch=000079" "epoch=000099" "epoch=000179" "epoch=000199")
for name in ${epochs_name[@]}
    do
    python eval/eval_clip_id.py \
        --video_dir "$log_dir/$name/samples/" \
        --text_dir "$log_dir/$name/input/" \
        --id_img_dir "datasets/benchmark_dataset/wearable_shoes2"
    done

log_dir="new_results/wearable_shoes2/2024-03-20T06-54-34_wearable_shoes2-lvdm-wrapper-shoes-dreamvideo"
epochs_name=("epoch=000039" "epoch=000079" "epoch=000099" "epoch=000179" "epoch=000199")
for name in ${epochs_name[@]}
    do
    python eval/eval_clip_id.py \
        --video_dir "$log_dir/$name/samples/" \
        --text_dir "$log_dir/$name/input/" \
        --id_img_dir "datasets/benchmark_dataset/wearable_shoes2"
    done

log_dir="new_results/wearable_shoes2/2024-03-18T16-03-03_wearable_shoes2-lvdm-wrapper-shoes-crossatten-ours-lr-0.00003-videocrafter2"
epochs_name=("epoch=000139" "epoch=000159" "epoch=000179" "epoch=000199")
for name in ${epochs_name[@]}
    do
    python eval/eval_clip_id.py \
        --video_dir "$log_dir/$name/samples/" \
        --text_dir "$log_dir/$name/input/" \
        --id_img_dir "datasets/benchmark_dataset/wearable_shoes2"
    done

log_dir="new_results/wearable_shoes2/2024-03-20T11-20-54_wearable_shoes2-lvdm-wrapper-shoes-crossatten-ours-lr-0.00003-videocrafter2"
epochs_name=("epoch=000139" "epoch=000159" "epoch=000179" "epoch=000199")
for name in ${epochs_name[@]}
    do
    python eval/eval_clip_id.py \
        --video_dir "$log_dir/$name/samples/" \
        --text_dir "$log_dir/$name/input/" \
        --id_img_dir "datasets/benchmark_dataset/wearable_shoes2"
    done

log_dir="new_results/wearable_shoes2/2024-03-20T11-20-54_wearable_shoes2-lvdm-wrapper-shoes-crossatten-ours-lr-0.00003-videocrafter2+_time_lora_0.4_0.8_5"
epochs_name=("epoch=000139" "epoch=000159" "epoch=000179" "epoch=000199")
for name in ${epochs_name[@]}
    do
    python eval/eval_clip_id.py \
        --video_dir "$log_dir/$name/samples/" \
        --text_dir "$log_dir/$name/input/" \
        --id_img_dir "datasets/benchmark_dataset/wearable_shoes2"
    done