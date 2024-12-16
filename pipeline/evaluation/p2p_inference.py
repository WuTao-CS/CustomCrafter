import argparse, os, sys, glob, yaml, math, random
import datetime, time
import numpy as np
from omegaconf import OmegaConf
from tqdm import trange, tqdm
from einops import repeat
from collections import OrderedDict
from transformers import CLIPTokenizer, CLIPTextModel
import torch
import torchvision
from torch import Tensor
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from lvdm.models.samplers.ddim import DDIMSampler
from utils.utils import instantiate_from_config
from lvdm.utils.ptp_utils import create_controller

# def load_model_checkpoint(model, ckpt):
#     print(f"Loading model from {ckpt}")
#     pl_sd = torch.load(ckpt, map_location="cpu")
#     try:
#         if 'state_dict' in pl_sd.keys():
#             model.load_state_dict(pl_sd["state_dict"],strict=True)
#             del pl_sd 
#         else:       
#             # deepspeed
#             new_pl_sd = OrderedDict()
#             for key in pl_sd.keys():
#                 new_pl_sd[key[16:]]=pl_sd[key]

#             model.load_state_dict(new_pl_sd,strict=True)
#             del pl_sd 
#             del new_pl_sd
#     except:
#         model.load_state_dict(pl_sd)
#         del pl_sd 
#     return model

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

            model.load_state_dict(new_pl_sd,strict=False)
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
        

def inference_prompt(model, prompts, noise_shape, cross_attention_kwargs, n_samples=1, ddim_steps=50, ddim_eta=1., \
                unconditional_guidance_scale=1.0,**kwargs):
    ddim_sampler = DDIMSampler(model)

    batch_size = noise_shape[0]
    ## get condition embeddings (support single prompt and multi-prompts both)
    if isinstance(prompts, str):
        prompts = [prompts]
    if hasattr(model.cond_stage_model,"tokenizer"):
        tokenizer = model.cond_stage_model.tokenizer
    else:
        tokenizer = CLIPTokenizer.from_pretrained("/group/40034/jerryxwli/code/VideoCrafter_Share/checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K")
    controller = create_controller(
            prompts, cross_attention_kwargs=cross_attention_kwargs, num_inference_steps=ddim_steps, tokenizer=tokenizer, device=model.device)
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
        prompts = batch_size * [""]
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
                                            verbose=True,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            temporal_length=noise_shape[2],
                                            mask=cond_mask,
                                            controller=controller,
                                            x0=cond_z0, **kwargs
                                            )
            if idx == 0:
                episodes = samples
            else:
                ## b,c,t,h,w
                episodes = torch.cat([episodes, samples[:,:,n_condt:,:,:]], dim=2)

            ## update for next episode
            cond_mask = temporal_mask
        
        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(episodes)
        # batch_images = model.decode_first_stage_2DAE(episodes)
        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)

# , The full picture.medium shot
def run_inference(args, gpu_num, gpu_no, prompt_list=['a <new1> man is riding a bike, heading towards the camera. medium shot','a <new1> teddybear is riding a bike, heading towards the camera. medium shot'],
    cross_attention_kwargs = {
        "edit_type": "replace",
        "n_cross_replace": 0.2,
        "n_self_replace": 0.2,
        # "local_blend_words": ["cute", "black"]
        }):

    config = OmegaConf.load(args.base)
    data_config = config.pop("data", OmegaConf.create())
    model_config = config.pop("model", OmegaConf.create())
    model_config['params']['unet_config']['params']['use_checkpoint']=False   
    if args.lora and args.lora_scale is not None:
        model_config['params']['lora_scale'] = args.lora_scale         
    model = instantiate_from_config(model_config)
    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, 'checkpoints/model_512/model-001.ckpt', is_pretrained=True)
    if args.lora:
        model._inject_lora()
    model = load_model_checkpoint(model, args.ckpt_path)
    # model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()
    model = model.cuda(gpu_no)

    ## run over data
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.in_channels
    frames = model.temporal_length
    # frames = 64
    bs = len(prompt_list)
    noise_shape = [bs, channels, frames, h, w]

    realdir = os.path.join(args.savedir, "input")
    fakedir = os.path.join(args.savedir, "samples")
    os.makedirs(realdir, exist_ok=True)
    os.makedirs(fakedir, exist_ok=True)

    start = time.time()  
    ## prompt file setting
    
    num_samples = len(prompt_list)
    samples_split = num_samples // gpu_num
    prompts = prompt_list
    cross_attention_kwargs["n_cross_replace"] = args.n_cross_replace
    cross_attention_kwargs["n_self_replace"] = args.n_self_replace
    batch_samples = inference_prompt(model, prompts, noise_shape, cross_attention_kwargs, args.n_samples, args.ddim_steps, args.ddim_eta, \
                                args.unconditional_guidance_scale,
                                cond_tau=args.cond_tau, target_size=args.target_size)
    ## save each example individually
    for nn, samples in enumerate(batch_samples):
        ## samples : [n_samples,c,t,h,w]
        prompt = prompts[nn]
        filename = "%04d_randk%d_%d_%d"%(nn, gpu_no, int(args.n_self_replace*10), int(args.n_cross_replace*10))
        save_results(prompt, samples, None, filename, realdir, fakedir, fps=8)
    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--base", type=str, help="config (yaml) path")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--seed", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--cond_tau", type=float, default=1.0, help="",)
    parser.add_argument("--target_size", type=int, default=None, help="", nargs="+")
    parser.add_argument("--lora", type=bool, default=False, help="is use lora",)
    parser.add_argument("--n_cross_replace", type=float, default=0.2, help="",)
    parser.add_argument("--n_self_replace", type=float, default=0.2, help="",)
    parser.add_argument("--lora_scale", type=float, default=None, help="lora_scale",)
    return parser


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@CoLVDM Inference: %s"%now)
    parser = get_parser()
    args = parser.parse_args()
    prompts = ['a cute teddybear is jumping',
           'a black teddybear is jumping']
    cross_attention_kwargs = {
        "edit_type": "replace",
        "cross_replace_steps": 0.4,
        "self_replace_steps": 0.4,
        "local_blend_words": ["cute", "black"]
        }
    seed_everything(args.seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank, prompts, cross_attention_kwargs)