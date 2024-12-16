import torch
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import os
import sys
sys.path.append(os.getcwd())
import torch
from PIL import Image
from tqdm import tqdm
import argparse
import json
import numpy as np
from eval.eval_clip import read_prompt_from_txt, read_video_to_np
import glob
# def read_video_to_np(video_path):
#     vidreader = VideoReader(video_path, ctx=cpu(0))
#     vid_len = len(vidreader)
#     frames = vidreader.get_batch(list(range(vid_len))).asnumpy()
#     return frames # thwc

class DINOEvaluator(object):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu", vit_model='checkpoints/dino-vits16') -> None:
        self.device = device
        self.vit_model = vit_model
        self.model = ViTModel.from_pretrained(self.vit_model)
        self.preprocess = ViTFeatureExtractor.from_pretrained(self.vit_model)
        self.model.to(self.device)
        self.model.eval()
    '''
        ViTImageProcessor {
        "do_normalize": true,
        "do_rescale": true,
        "do_resize": true,
        "feature_extractor_type": "ViTFeatureExtractor",
        "image_mean": [
            0.485,
            0.456,
            0.406
        ],
        "image_processor_type": "ViTImageProcessor",
        "image_std": [
            0.229,
            0.224,
            0.225
        ],
        "resample": 2,
        "rescale_factor": 0.00392156862745098,
        "size": {
            "height": 224,
            "width": 224
        }
        }
            if do_normalize, then normalize the image with image_mean and image_std.
            [1, 1, 1] => [[2.249, 2.429, 2.64]]
    '''

    @torch.no_grad()
    def _encode_image(self, image) -> torch.Tensor:
        # print(image.shape)
        inputs  = self.preprocess(images=image, return_tensors="pt")
        inputs  = inputs.to(self.device)
        outputs = self.model(**inputs)
        # last_hidden_states: [B, 197, 384]
        last_hidden_states = outputs.last_hidden_state

        # We only use CLS token's features, so that the spatial location of the subject will not impact matching. 
        # [B, 384]
        return last_hidden_states[:, 0]

    def get_image_features(self, image, norm: bool = True) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif isinstance(image, str):
            image = Image.open(image).convert('RGB')
        image = np.array(image)
        image_features = self._encode_image(image)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features
    @torch.no_grad()
    def _cal_sim(self, feat1, feat2):
        similarity = (feat1 @ feat2.T).mean()
        return similarity
    @torch.no_grad()
    def cal_video_id_consistency(self, video_path, id_img_path, stride=1):
        video = read_video_to_np(video_path) if isinstance(video_path, str) else video_path
        num_frames = video.shape[0]
        sim_total, c = 0, 0
        feat_id = self.get_image_features(id_img_path)
        
        for i in range(0, num_frames, stride):
            feat_frame = self.get_image_features(video[i, ...])
            sim_total += self._cal_sim(feat_id, feat_frame)
            c += 1
        return sim_total / c
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default=None, help="generated videos dir")
    parser.add_argument("--id_img_dir", type=str, default=None, help="id images dir")
    parser.add_argument("--text_dir", type=str, default=None, help="text prompts dir")
    parser.add_argument("--save_dir", type=str, default=None, help="result saving dir")
    parser.add_argument("--verbose", default=False, action='store_true')
    args = parser.parse_args()

    video_dir = args.video_dir
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
    
    Dinoeval = DINOEvaluator()

    score_id_total, count_id = 0, 0
    for vidpath in tqdm(video_paths):
        video = read_video_to_np(vidpath)
        for id_path in id_paths:
            score_id_total += Dinoeval.cal_video_id_consistency(video, id_path)
            count_id += 1

    id = (score_id_total / count_id).item()

    res = {
           'Dino_id_consistency': id,
           }
    
    print(res)
    f = open(os.path.join(save_dir, f'Dino_results_id.json'), 'w')
    json.dump(res, f)
    f.close()