"""Inception Score (IS) from the paper "Improved techniques for training
GANs". Matches the original implementation by Salimans et al. at
https://github.com/openai/improved-gan/blob/master/inception_score/model.py"""

from decord import VideoReader, cpu
from urllib.parse import urlparse
from eval import dnnlib
from tqdm import tqdm
from PIL import Image

import torch.nn as nn
import numpy as np
import argparse
import pickle
import torch
import tqdm
import clip
import os


@torch.no_grad()
def inception_score(model_url, device, paths, splits=1, verbose=False):
    with dnnlib.util.open_url(model_url, verbose=verbose) as f:
        if urlparse(model_url).path.endswith('.pkl'):
            model = pickle.load(f).to(device)
        else:
            model = torch.jit.load(f).eval().to(device)

    features, split_scores = list(), list()
    for path in tqdm.tqdm(paths):
        video = torch.tensor(read_video_to_np(path))  # t h w c
        video = video.permute(3, 0, 1, 2).contiguous()  # c t h w

        if video.shape[1] == 1:
            video = torch.repeat_interleave(video, 3, dim=1)
        features.append(model(video.to(device).unsqueeze(0)).cpu().numpy())
    features = np.concatenate(features, axis=0)
    num_samples = features.shape[0]

    for k in range(splits):
        part = features[k * (num_samples // splits): (k + 1) * (num_samples // splits), :]
        kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
        kl = np.mean(np.sum(kl, axis=1))
        split_scores.append(np.exp(kl))
    return float(np.mean(split_scores)), float(np.std(split_scores))


def read_video_to_np(video_path):
    vidreader = VideoReader(video_path, ctx=cpu(0))
    vid_len = len(vidreader)
    frames = vidreader.get_batch(list(range(vid_len))).asnumpy()
    return frames


def read_prompt_from_txt(fpath):
    prompt_list = []
    with open(fpath, 'r') as f:
        for l in f.readlines():
            l = l.strip()
            if len(l) != 0:
                prompt_list.append(l)
    assert (len(prompt_list) == 1), len(prompt_list)
    return prompt_list[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expdir", type=str, default=None, help="results saving path")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--splits", type=int, default=1)
    args = parser.parse_args()

    video_paths = [os.path.join(args.expdir, f) for f in os.listdir(args.expdir)]
    device = torch.device('cuda')

    mean, std = inception_score(args.model_dir, device, video_paths, args.splits, verbose=False)
    print(f"Inception score: {mean}, std: {std}.")


if __name__ == "__main__":
    main()
