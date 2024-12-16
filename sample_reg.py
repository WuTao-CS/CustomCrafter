
import argparse, os, sys, glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a photo of a cat",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=200
    )

    opt = parser.parse_args()
    outpath = opt.outdir
    os.makedirs(outpath, exist_ok=True)
    sample_path = os.path.join(outpath, "images")

    os.makedirs(sample_path, exist_ok=True)
    prompt = opt.prompt

    num_sample=opt.num

    repo_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    batch_size = 4
    num_batch = int(num_sample / batch_size)
    cnt = 0
    captions=[prompt]*num_sample
    paths=[]
    for i in range(num_batch):
        images = pipe(prompt, num_inference_steps=50, num_images_per_prompt=batch_size).images
        for j in range(batch_size):
            image = images[j]
            image.save(os.path.join(sample_path,"{}.png".format(cnt)))
            paths.append(os.path.join(sample_path,"{}.png".format(cnt)))
            cnt+=1
    with open(f'{outpath}/caption.txt', 'w') as f:
        for each in captions:
            f.write(each.strip() + '\n')

    with open(f'{outpath}/images.txt', 'w') as f:
        for each in paths:
            f.write(each.strip() + '\n')

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
