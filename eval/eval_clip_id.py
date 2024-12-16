import os
import sys
sys.path.append(os.getcwd())
import torch
import clip
from PIL import Image
from decord import VideoReader, cpu
from tqdm import tqdm
import argparse
import json
from eval.eval_clip import read_prompt_from_txt, read_video_to_np, ClipEval
from eval.eval_dino import DINOEvaluator

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default=None, help="generated videos dir")
    parser.add_argument("--text_dir", type=str, default=None, help="text prompts dir")
    parser.add_argument("--id_img_dir", type=str, default=None, help="id images dir")
    parser.add_argument("--save_dir", type=str, default=None, help="result saving dir")
    parser.add_argument("--verbose", default=False, action='store_true')
    args = parser.parse_args()

    video_dir = args.video_dir
    text_dir = args.text_dir
    if args.save_dir is None:
        save_dir = video_dir
    else:
        save_dir = args.save_dir
    id_img_dir = args.id_img_dir
    os.makedirs(save_dir, exist_ok=True)

    video_paths=[]
    for f in sorted(os.listdir(video_dir)):
        if f.endswith(".mp4"):
            video_paths.append(os.path.join(video_dir, f))
    id_paths = [os.path.join(id_img_dir, f) for f in sorted(os.listdir(id_img_dir))]
    texts = [read_prompt_from_txt(os.path.join(text_dir, fn)) for fn in sorted(os.listdir(text_dir))]
    
    clipeval = ClipEval()
    Dinoeval = DINOEvaluator()

    score_vta_total, score_fc_total, count = 0, 0, 0
    score_id_total, count_id = 0, 0
    dino_id_total = 0
    for vidpath, txt in tqdm(zip(video_paths, texts), desc='Video', total=min(len(video_paths), len(texts))):
        video = read_video_to_np(vidpath)
        score_vta = clipeval.cal_video_text_alignment(video, txt, verbose=args.verbose)
        score_fc = clipeval.cal_video_frame_consistency(video, verbose=args.verbose)
        for id_path in id_paths:
            score_id_total += clipeval.cal_video_id_consistency(video, id_path)
            dino_id_total += Dinoeval.cal_video_id_consistency(video, id_path).item()
            count_id += 1

        score_vta_total += score_vta
        score_fc_total += score_fc
        count += 1
    vta = score_vta_total / count
    fc = score_fc_total / count
    id = score_id_total / count_id
    dino_id = dino_id_total / count_id
    res = {'clip_t': vta,
           'video_frame_consistency': fc,
           'clip_i': id,
           'dino_i': dino_id,
           }
    
    print(res)
    f = open(os.path.join(save_dir, f'results_id.json'), 'w')
    json.dump(res, f)
    f.close()


