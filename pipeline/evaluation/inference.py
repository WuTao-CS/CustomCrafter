import argparse, os, sys, glob, yaml, math, random
import datetime, time
import numpy as np
from omegaconf import OmegaConf
from tqdm import trange, tqdm
from einops import repeat
from collections import OrderedDict

import torch
import torchvision
from torch import Tensor
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from lvdm.models.samplers.ddim import DDIMSampler
from utils.utils import instantiate_from_config


def load_model_checkpoint(model, ckpt, is_pretrained=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if is_pretrained:
        del pl_sd["state_dict"]["cond_stage_model.model.token_embedding.weight"]
    try:
        if 'state_dict' in pl_sd.keys():
            m, u = model.load_state_dict(pl_sd["state_dict"],strict=False)
            if is_pretrained:
                print("Missing keys:", m)
                # print("Unexpected keys:", u)
            else:
                # print("Missing keys:", m)
                print("Unexpected keys:", u)
            
            del pl_sd 
        else:       
            # deepspeed
            new_pl_sd = OrderedDict()
            for key in pl_sd.keys():
                new_pl_sd[key[16:]]=pl_sd[key]

            m, u = model.load_state_dict(new_pl_sd,strict=False)
            print("Missing keys:", m)
            print("Unexpected keys:", u)
            del pl_sd 
            del new_pl_sd
    except:
        model.load_state_dict(pl_sd)
        del pl_sd 
    return model

def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list

def save_results(prompt, samples, inputs, filename, realdir, fakedir, fps=10):
    ## save prompt
    prompt = prompt[0] if isinstance(prompt, list) else prompt
    path = os.path.join(realdir, "%s.txt"%filename)
    with open(path, 'w') as f:
        f.write(f'{prompt}')
        f.close()

    ## save video
    videos = [inputs, samples]
    savedirs = [realdir, fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        # b,c,t,h,w
        video = video.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n)) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        path = os.path.join(savedirs[idx], "%s.mp4"%filename)
        torchvision.io.write_video(path, grid, fps=fps, video_codec='h264', options={'crf': '10'})
        

def inference_prompt(model, prompts, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., \
                unconditional_guidance_scale=1.0, **kwargs):
    ddim_sampler = DDIMSampler(model)

    batch_size = noise_shape[0]
    ## get condition embeddings (support single prompt and multi-prompts both)
    if isinstance(prompts, str):
        prompts = [prompts]
    cond_list = []
    n_episodes = 1
    for prompt_idx in prompts:
        ## split prompts (if multi-prompts used for each sample)
        sub_prompts = prompt_idx.split('&&')   
        n_episodes = len(sub_prompts)
        cond_list.extend(sub_prompts)
    cond_episodes = []
    for episode in range(n_episodes):
        batch_cond = []
        for i in range(batch_size):
            batch_cond.append(cond_list[i*n_episodes+episode])
        batch_cond = model.get_learned_conditioning(batch_cond)
        cond_episodes.append(batch_cond)

    if unconditional_guidance_scale != 1.0:
        prompts = batch_size * ["blurry background, blurry picture, Stationary, low-motion, deformed, lowres, bad anatomy, text, error, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"]
        uc = model.get_learned_conditioning(prompts)
    else:
        uc = None
    
    ## for multi-prompts only
    n_condt = 4
    temporal_len = noise_shape[2]
    temporal_mask = torch.zeros(noise_shape[2], device=model.device)
    temporal_mask[:n_condt] = 1.0
    ## b,c,t,h,w
    temporal_mask = temporal_mask[None,None,:,None,None]

    batch_variants = []
    for _ in range(n_samples):
        episodes = None
        cond_mask = None
        # cond_z0 = torch.zeros(noise_shape, device=model.device)
        cond_z0 = None
        for idx, cond in enumerate(cond_episodes):
            kwargs.update({"clean_cond": False})
            samples, intermediates = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=noise_shape[0],
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            temporal_length=noise_shape[2],
                                            mask=cond_mask,
                                            x0=cond_z0, **kwargs
                                            )
            if idx == 0:
                episodes = samples
            else:
                ## b,c,t,h,w
                episodes = torch.cat([episodes, samples[:,:,n_condt:,:,:]], dim=2)

            ## update for next episode
            cond_mask = temporal_mask
            # cond_z0[:,:,:n_condt,:,:] = samples[:,:,(temporal_len-n_condt):,:,:]
        
        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(episodes)
        # batch_images = model.decode_first_stage_2DAE(episodes)
        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)


def run_inference(args, gpu_num, gpu_no):
    ## model config
    config = OmegaConf.load(args.base)
    data_config = config.pop("data", OmegaConf.create())
    model_config = config.pop("model", OmegaConf.create())
    model_config['params']['unet_config']['params']['use_checkpoint']=False        
    if args.lora and args.lora_scale is not None:
        model_config['params']['lora_scale'] = args.lora_scale
    model = instantiate_from_config(model_config)
    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, args.pretrain, is_pretrained=True)
    if args.lora:
        model._inject_lora()
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()
    model = model.cuda(gpu_no)

    ## run over data
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.in_channels
    frames = model.temporal_length
    # frames = 64
    noise_shape = [args.bs, channels, frames, h, w]

    realdir = os.path.join(args.savedir, "input")
    fakedir = os.path.join(args.savedir, "samples")
    os.makedirs(realdir, exist_ok=True)
    os.makedirs(fakedir, exist_ok=True)

    start = time.time()  
    if args.prompt_file:
        ## prompt file setting
        assert os.path.exists(args.prompt_file), "Error: prompt file Not Found!"
        prompt_list = load_prompts(args.prompt_file)
        num_samples = len(prompt_list)
        samples_split = num_samples // gpu_num
        print('Prompts testing [rank:%d] %d/%d samples loaded.'%(gpu_no, samples_split, num_samples))
        #indices = random.choices(list(range(0, num_samples)), k=samples_per_device)
        indices = list(range(samples_split*gpu_no, samples_split*(gpu_no+1)))
        prompt_list_rank = [prompt_list[i] for i in indices]

        for idx, indice in tqdm(enumerate(range(0, len(prompt_list_rank), args.bs)), desc='Sample Batch'):
            prompts = prompt_list_rank[indice:indice+args.bs]
            batch_samples = inference_prompt(model, prompts, noise_shape, args.n_samples, args.ddim_steps, args.ddim_eta, \
                                        args.unconditional_guidance_scale,
                                        cond_tau=args.cond_tau, target_size=args.target_size)
            ## save each example individually
            for nn, samples in enumerate(batch_samples):
                ## samples : [n_samples,c,t,h,w]
                prompt = prompts[nn]
                filename = "%04d_randk%d"%(idx*args.bs+nn+gpu_no*len(prompt_list_rank), gpu_no)
                save_results(prompt, samples, None, filename, realdir, fakedir, fps=8)
    else:
        ## dataset settting
        try:
            dataset = instantiate_from_config(data_config.params.validation)
        except:
            print("Error: dataset configure failed!")

        num_samples = len(dataset)
        samples_split = num_samples // gpu_num
        print('Dataset testing [rank:%d] %d/%d samples loaded.'%(gpu_no, samples_split, num_samples))
        #indices = random.choices(list(range(0, num_samples)), k=samples_split)
        indices = list(range(samples_split*gpu_no, samples_split*(gpu_no+1)))
        dataset_rank = torch.utils.data.Subset(dataset, indices)
        dataloader_rank = DataLoader(dataset_rank, batch_size=args.bs, num_workers=4, shuffle=False)

        for idx, batch in tqdm(enumerate(dataloader_rank), desc='Sample Batch'):
            prompts = batch[model.cond_stage_key]
            batch_samples = inference_prompt(model, prompts, noise_shape, args.n_samples, args.ddim_steps, args.ddim_eta, \
                                        args.unconditional_guidance_scale, 
                                        cond_tau=args.cond_tau, target_size=args.target_size)
            ## save each example individually
            inputs = batch[model.first_stage_key]
            inputs = inputs.unsqueeze(0).permute(1,0,2,3,4,5)
            for nn, samples in enumerate(batch_samples):
                ## samples : [n_samples,c,t,h,w]
                prompt = prompts[nn]
                filename = "%04d_randk%d"%(idx*args.bs+nn, gpu_no)
                save_results(prompt, samples, inputs[nn], filename, realdir, fakedir, fps=10)
    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--base", type=str, help="config (yaml) path")
    parser.add_argument("--prompt_file", type=str, default=None, help="a text file containing many prompts")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--seed", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--cond_tau", type=float, default=1.0, help="",)
    parser.add_argument("--target_size", type=int, default=None, help="", nargs="+")
    parser.add_argument("--lora", type=bool, default=False, help="is use lora",)
    parser.add_argument("--lora_scale", type=float, default=None, help="lora_scale",)
    parser.add_argument("--pretrain", type=str, default='checkpoints/model_512/model-001.ckpt', help="pretrain model path")
    return parser


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@CoLVDM Inference: %s"%now)
    parser = get_parser()
    args = parser.parse_args()

    seed_everything(args.seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)