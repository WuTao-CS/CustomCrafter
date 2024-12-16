import os
import numpy as np
import torch
import clip
from PIL import Image
from decord import VideoReader, cpu
from tqdm import tqdm
import argparse
import json

def read_video_to_np(video_path):
    vidreader = VideoReader(video_path, ctx=cpu(0))
    vid_len = len(vidreader)
    frames = vidreader.get_batch(list(range(vid_len))).asnumpy()
    return frames # thwc

def read_prompt_from_txt(fpath):
    prompt_list = []
    with open(fpath, 'r') as f:
        for l in f.readlines():
            l = l.strip()
            if len(l) != 0:
                prompt_list.append(l)
    assert(len(prompt_list) == 1), len(prompt_list)
    return prompt_list[0]

class ClipEval():
    def __init__(self, 
                 device= "cuda" if torch.cuda.is_available() else "cpu"
                 ) -> None:
        model, preprocess = clip.load("checkpoints/clip/ViT-B-32.pt", device=device)
        self.model = model
        self.preprocess = preprocess
        self.device = device
    
    @torch.no_grad()
    def _encode_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, str):
            image = Image.open(image)
        image = self.preprocess(image).unsqueeze(0).to(self.device) # image:hwc np
        image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    @torch.no_grad()
    def _encode_text(self, text):
        text = clip.tokenize(text).to(self.device)
        text_features = self.model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    @torch.no_grad()
    def _cal_sim(self, feat1, feat2):
        similarity = torch.cosine_similarity(feat1, feat2, dim=-1).item()
        return similarity
    
    @torch.no_grad()
    def cal_prob(self, image, texts):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, str):
            image = Image.open(image)
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        texts = clip.tokenize(texts).to(self.device)
        logits_per_image, logits_per_text = self.model(image, texts)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return probs

    @torch.no_grad()
    def cal_image_text_alignment(self, img_path, text):
        img_feat = self._encode_image(Image.open(img_path))
        txt_feat = self._encode_text(text)
        return self._cal_sim(img_feat, txt_feat)

    @torch.no_grad()
    def cal_video_text_alignment(self, video_path, text, verbose=False):
        video = read_video_to_np(video_path) if isinstance(video_path, str) else video_path
        sim_total, c = 0, 0
        
        txt_feat = self._encode_text(text)
        for frame in video: # thwc
            img_feat = self._encode_image(frame)
            sim_total += self._cal_sim(img_feat, txt_feat)
            c += 1
        avg = sim_total / c
        if verbose:
            print(f'video-text similarity={avg}')
        return avg
    
    @torch.no_grad()
    def cal_video_frame_consistency(self, video_path, verbose=False):
        video = read_video_to_np(video_path) if isinstance(video_path, str) else video_path
        num_frames = video.shape[0]
        sim_total, c = 0, 0
        for i in range(num_frames-1):
            feat1, feat2 = self._encode_image(video[i, ...]),  self._encode_image(video[i+1, ...])
            sim_total += self._cal_sim(feat1, feat2)
            c += 1
        avg = sim_total / c
        if verbose:
            print(f'video frame consistency={avg}')
        return avg
    
    @torch.no_grad()
    def cal_image_id_consistency(self, img_path, id_img_path):
        img = Image.open(img_path)
        id_img = Image.open(id_img_path)
        feat1, feat2 = self._encode_image(img), self._encode_image(id_img)
        return self._cal_sim(feat1, feat2)
    
    @torch.no_grad()
    def cal_video_id_consistency(self, video_path, id_img_path, stride=1):
        video = read_video_to_np(video_path) if isinstance(video_path, str) else video_path
        num_frames = video.shape[0]
        sim_total, c = 0, 0
        feat_id = self._encode_image(id_img_path)
        for i in range(0, num_frames, stride):
            feat_frame = self._encode_image(video[i, ...])
            sim_total += self._cal_sim(feat_id, feat_frame)
            c += 1
        return sim_total / c

                
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default=None, help="generated videos saving path")
    parser.add_argument("--text_dir", type=str, default=None, help="text prompts saving path")
    parser.add_argument("--save_dir", type=str, default=None, help="text prompts saving path")
    parser.add_argument("--verbose", default=False, action='store_true')
    args = parser.parse_args()

    video_dir = args.video_dir
    text_dir = args.text_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    video_paths = [os.path.join(video_dir, f) for f in sorted(os.listdir(video_dir))]
    texts = [read_prompt_from_txt(os.path.join(text_dir, fn)) for fn in sorted(os.listdir(text_dir))]
    
    clipeval = ClipEval()

    score_vta_total, score_fc_total, count = 0, 0, 0
    for vidpath, txt in tqdm(zip(video_paths, texts), desc='Video', total=min(len(video_paths), len(texts))):
        video = read_video_to_np(vidpath)
        score_vta = clipeval.cal_video_text_alignment(video, txt, verbose=args.verbose)
        score_fc = clipeval.cal_video_frame_consistency(video, verbose=args.verbose)
        score_vta_total += score_vta
        score_fc_total += score_fc
        count += 1
    vta = score_vta_total / count
    fc = score_fc_total / count

    res = {'video_text_alignment': vta,
           'video_frame_consistency': fc
           }
    
    print(res)
    f = open(os.path.join(save_dir, f'results_clip.json'), 'w')
    json.dump(res, f)
    f.close()


