import os, random
from functools import partial
from contextlib import contextmanager
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat
import itertools
import logging
mainlogger = logging.getLogger('mainlogger')

import torch
import torch.nn as nn
from torch import einsum
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
from transformers import CLIPVisionModel
from utils.utils import instantiate_from_config, count_params, check_istarget
from lvdm.ema import LitEma
from lvdm.distributions import normal_kl, DiagonalGaussianDistribution
from lvdm.models.utils_diffusion import make_beta_schedule
from lvdm.modules.encoders.ip_resampler import ImageProjModel, Resampler
from lvdm.models.samplers.ddim import DDIMSampler
from utils.save_video import log_local, prepare_to_log, log_txt_as_img
from utils.cal_fvd_text2video import cal_fvd
from lvdm.basics import disabled_train
from lvdm.common import (
    extract_into_tensor,
    noise_like,
    exists,
    default
)
from lvdm.models.ddpm3d import LatentDiffusion, CustomDiffusion
from lvdm.models.lora import LoraInjectedLinear
import lvdm.models.lora as lora


class LoRaCrossAttenDiffusion(CustomDiffusion):
    def __init__(self,
                #  ckpt_path, # ckpt path for pretrained base model
                 lora_rank,
                 inject_unet,
                 inject_clip,
                 inject_unet_key_word,
                 inject_clip_key_word,
                 lora_scale=1.0,
                 inject_unet_child_name=None,
                 update_keys=None,
                 inject_init_attn=True,
                 inject_ffn=False,
                 only_attn2=False,
                 inject_ta=False,
                 *args, **kwargs
                 ):
        # init & load pretrained
        super().__init__(*args, **kwargs)
        # super().__init__(ckpt_path=ckpt_path, *args, **kwargs)
        
        self.lora_scale = lora_scale
        self.lora_rank = lora_rank
        self.inject_unet = inject_unet
        self.inject_clip = inject_clip
        self.inject_unet_key_word = inject_unet_key_word
        self.inject_clip_key_word = inject_clip_key_word
        self.inject_unet_child_name = inject_unet_child_name
        self.update_keys = update_keys
        self.inject_init_attn = inject_init_attn
        self.inject_ffn = inject_ffn
        self.inject_ta = inject_ta
        self.only_attn2 = only_attn2
        self._freeze_model()
        # self._inject_lora()

    def on_save_checkpoint(self, checkpoint):
        # pop the backbone here using custom logic
        learnable_params = self.get_learnable_params()
        for key in list(checkpoint['state_dict'].keys()):
            if key not in learnable_params:
                checkpoint['state_dict'].pop(key)
        return checkpoint
    
    def get_learnable_params(self):
        """ get learnable params """
        params = []
        for name, para in self.named_parameters():
            if para.requires_grad:
                params.append(name)
        return params

    def _freeze_model(self):
        for x in self.model.diffusion_model.named_parameters():
                x[1].requires_grad = False

    def _inject_lora(self,):
        """
        inject lora to unet and text encoder
        """
        if self.inject_unet:
            self.lora_require_grad_params, self.lora_names = lora.inject_trainable_lora(self.model, 
                                                                                        self.inject_unet_key_word, 
                                                                                        r=self.lora_rank,
                                                                                        scale=self.lora_scale,
                                                                                        verbose=True,
                                                                                        module_child_name=self.inject_unet_child_name,
                                                                                        inject_init_attn=self.inject_init_attn,
                                                                                        inject_ffn=self.inject_ffn,
                                                                                        only_attn2=self.only_attn2,
                                                                                        inject_ta=self.inject_ta,
                                                                                        )
        if self.inject_clip:
            self.lora_require_grad_params_clip, self.lora_names_clip = lora.inject_trainable_lora(self.cond_stage_model, 
                                                                                                  self.inject_clip_key_word, 
                                                                                                  r=self.lora_rank,
                                                                                                  scale=self.lora_scale
                                                                                                )

    def configure_optimizers(self):
        """ configure_optimizers for LatentDiffusion + lora """
        ## update lora parameters
        if self.inject_unet:
            params = list(filter(lambda p: p.requires_grad, self.model.diffusion_model.parameters()))
        if self.inject_clip:
            if self.inject_unet:
                params = params + list(itertools.chain(*self.lora_require_grad_params_clip))
            else:
                params = list(itertools.chain(*self.lora_require_grad_params_clip))
        ### add target text token 
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            if self.add_token:
                if self.text_model_type == "clip":
                    params = params + list(self.cond_stage_model.transformer.text_model.embeddings.token_embedding.parameters())
                elif self.text_model_type == "openclip":
                    params = params + list(self.cond_stage_model.model.token_embedding.parameters()) # [torch.Size([49409, 1024])]
            else:
                params = params + list(self.cond_stage_model.parameters())
        
        if self.learn_logvar:
            mainlogger.info('Diffusion model optimizing logvar')
            if isinstance(params[0], dict):
                params.append({"params": [self.logvar]})
            else:
                params.append(self.logvar)
        
        if self.update_keys is not None:
            params_trained = []
            for n, p in self.model.named_parameters():
                print(f'n={n}')
                if n in self.update_keys:
                    print(f"Also update parameter: {n}")
                    # params_trained.append(p)
                    params.append(p)
            # params = [{'params': params}, 
            #           {'params': params_trained, 'lr': self.learning_rate/50},
            #           ]
        ## optimizer
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
        
        ## lr scheduler
        if self.use_scheduler:
            mainlogger.info("Setting up LambdaLR scheduler...")
            lr_scheduler = self.configure_schedulers(optimizer)
            return [optimizer], [lr_scheduler]
        
        return optimizer


class LoRatestDiffusion(LatentDiffusion):
    def __init__(self,
                #  ckpt_path, # ckpt path for pretrained base model
                 lora_rank,
                 inject_unet,
                 inject_clip,
                 inject_unet_key_word,
                 inject_clip_key_word,
                 lora_scale=1.0,
                 inject_unet_child_name=None,
                 update_keys=None,
                 inject_init_attn=True,
                 inject_ffn=False,
                 only_attn2=False,
                 *args, **kwargs
                 ):
        # init & load pretrained
        super().__init__(*args, **kwargs)
        # super().__init__(ckpt_path=ckpt_path, *args, **kwargs)
        
        self.lora_scale = lora_scale
        self.lora_rank = lora_rank
        self.inject_unet = inject_unet
        self.inject_clip = inject_clip
        self.inject_unet_key_word = inject_unet_key_word
        self.inject_clip_key_word = inject_clip_key_word
        self.inject_unet_child_name = inject_unet_child_name
        self.update_keys = update_keys
        self.inject_init_attn = inject_init_attn
        self.inject_ffn = inject_ffn
        self.only_attn2 = only_attn2
        self._freeze_model()
        # self._inject_lora()

    def _freeze_model(self):
        for name, para in self.model.diffusion_model.named_parameters():
            para.requires_grad = False
    def reset_lora_scale(self, new_scale):
        print("change lora scale to :",new_scale)
        def change_scale(model, new_scale):
            for layer in model.children():
                if type(layer) == LoraInjectedLinear:
                    setattr(layer, 'scale', new_scale)
                else:
                    change_scale(layer, new_scale)
        change_scale(self.model.diffusion_model, new_scale)
    def _inject_lora(self,):
        """
        inject lora to unet and text encoder
        """
        print(self.lora_scale)
        if self.inject_unet:
            self.lora_require_grad_params, self.lora_names = lora.inject_trainable_lora(self.model, 
                                                                                        self.inject_unet_key_word, 
                                                                                        r=self.lora_rank,
                                                                                        scale=self.lora_scale,
                                                                                        verbose=True,
                                                                                        module_child_name=self.inject_unet_child_name,
                                                                                        inject_init_attn = self.inject_init_attn,
                                                                                        inject_ffn = self.inject_ffn,
                                                                                        only_attn2 = self.only_attn2
                                                                                        )
        if self.inject_clip:
            self.lora_require_grad_params_clip, self.lora_names_clip = lora.inject_trainable_lora(self.cond_stage_model, 
                                                                                                  self.inject_clip_key_word, 
                                                                                                  r=self.lora_rank,
                                                                                                  scale=self.lora_scale
                                                                                                )

    def configure_optimizers(self):
        """ configure_optimizers for LatentDiffusion + lora """
        ## update lora parameters
        if self.inject_unet:
            params = list(itertools.chain(*self.lora_require_grad_params))
        if self.inject_clip:
            if self.inject_unet:
                params = params+list(itertools.chain(*self.lora_require_grad_params_clip))
            else:
                params = list(itertools.chain(*self.lora_require_grad_params_clip))
        
        
        ### add target text token 
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            if self.add_token:
                if self.text_model_type == "clip":
                    params = params + list(self.cond_stage_model.transformer.text_model.embeddings.token_embedding.parameters())
                elif self.text_model_type == "openclip":
                    params = params + list(self.cond_stage_model.model.token_embedding.parameters()) # [torch.Size([49409, 1024])]
            else:
                params = params + list(self.cond_stage_model.parameters())
        
        if self.learn_logvar:
            mainlogger.info('Diffusion model optimizing logvar')
            if isinstance(params[0], dict):
                params.append({"params": [self.logvar]})
            else:
                params.append(self.logvar)
        
        if self.update_keys is not None:
            params_trained = []
            for n, p in self.model.named_parameters():
                print(f'n={n}')
                if n in self.update_keys:
                    print(f"Also update parameter: {n}")
                    # params_trained.append(p)
                    params.append(p)
            # params = [{'params': params}, 
            #           {'params': params_trained, 'lr': self.learning_rate/50},
            #           ]
        ## optimizer
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
        
        ## lr scheduler
        if self.use_scheduler:
            mainlogger.info("Setting up LambdaLR scheduler...")
            lr_scheduler = self.configure_schedulers(optimizer)
            return [optimizer], [lr_scheduler]
        
        return optimizer



class Mapper(nn.Module):
    def __init__(self,
        input_dim: int,
        output_dim: int,
    ):
        super(Mapper, self).__init__()

        for i in range(5):
            setattr(self, f'mapping_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                         nn.LayerNorm(1024),
                                         nn.LeakyReLU(),
                                         nn.Linear(1024, 1024),
                                         nn.LayerNorm(1024),
                                         nn.LeakyReLU(),
                                         nn.Linear(1024, output_dim)))

            setattr(self, f'mapping_patch_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, output_dim)))

    def forward(self, embs):
        hidden_states = ()
        for i, emb in enumerate(embs):
            hidden_state = getattr(self, f'mapping_{i}')(emb[:, :1]) + getattr(self, f'mapping_patch_{i}')(emb[:, 1:]).mean(dim=1, keepdim=True)
            hidden_states += (hidden_state, )
        hidden_states = torch.cat(hidden_states, dim=1)
        return hidden_states
    
def freeze_params(params):
    for param in params:
        param.requires_grad = False
class LoRaVisualCustomDiffusion(CustomDiffusion):
    def __init__(self,
                #  ckpt_path, # ckpt path for pretrained base model
                 lora_rank,
                 inject_unet,
                 inject_clip,
                 inject_unet_key_word,
                 inject_clip_key_word,
                 lora_scale=1.0,
                 inject_unet_child_name=None,
                 update_keys=None,
                 inject_init_attn=True,
                 inject_ffn=False,
                 only_attn2=False,

                 *args, **kwargs
                 ):
        # init & load pretrained
        super().__init__(*args, **kwargs)
        # super().__init__(ckpt_path=ckpt_path, *args, **kwargs)
        self.mapper = Mapper(input_dim=1280, output_dim=1024)
        
        self.lora_scale = lora_scale
        self.lora_rank = lora_rank
        self.inject_unet = inject_unet
        self.inject_clip = inject_clip
        self.inject_unet_key_word = inject_unet_key_word
        self.inject_clip_key_word = inject_clip_key_word
        self.inject_unet_child_name = inject_unet_child_name
        self.update_keys = update_keys
        self.inject_init_attn = inject_init_attn
        self.inject_ffn = inject_ffn
        self.only_attn2 = only_attn2
        self._freeze_model()
        # self._inject_lora()
    def on_save_checkpoint(self, checkpoint):
        # pop the backbone here using custom logic
        learnable_params = self.get_learnable_params()
        for key in list(checkpoint['state_dict'].keys()):
            if key not in learnable_params:
                checkpoint['state_dict'].pop(key)
        return checkpoint
    
    def get_learnable_params(self):
        """ get learnable params """
        params = []
        for name, para in self.named_parameters():
            if para.requires_grad:
                params.append(name)
        return params

    def _freeze_model(self):
        if self.freeze_model == 'crossattn':
            for x in self.model.diffusion_model.named_parameters():
                if 'transformer_blocks' not in x[0]:
                    x[1].requires_grad = False
                elif 'init_attn' in x[0]:
                    x[1].requires_grad = False
                elif not ('attn2.to_q' in x[0] or 'attn2.to_k' in x[0] or 'attn2.to_v' in x[0]):
                    x[1].requires_grad = False
                else:
                    x[1].requires_grad = False
        elif self.freeze_model == 'crossattn-ffn':
            for x in self.model.diffusion_model.named_parameters():
                if 'transformer_blocks' not in x[0]:
                    x[1].requires_grad = False
                elif 'init_attn' in x[0]:
                    x[1].requires_grad = False
                elif '.ff.' in x[0]:
                    x[1].requires_grad = True
                else:
                    x[1].requires_grad = False
        else:
            raise NotImplementedError
        for param in self.mapper.parameters():
            param.requires_grad = True

    def _inject_lora(self,):
        """
        inject lora to unet and text encoder
        """
        if self.inject_unet:
            self.lora_require_grad_params, self.lora_names = lora.inject_trainable_lora(self.model, 
                                                                                        self.inject_unet_key_word, 
                                                                                        r=self.lora_rank,
                                                                                        scale=self.lora_scale,
                                                                                        verbose=True,
                                                                                        module_child_name=self.inject_unet_child_name,
                                                                                        inject_init_attn=self.inject_init_attn,
                                                                                        inject_ffn=self.inject_ffn,
                                                                                        only_attn2=self.only_attn2
                                                                                        )
        if self.inject_clip:
            self.lora_require_grad_params_clip, self.lora_names_clip = lora.inject_trainable_lora(self.cond_stage_model, 
                                                                                                  self.inject_clip_key_word, 
                                                                                                  r=self.lora_rank,
                                                                                                  scale=self.lora_scale
                                                                                                )

    def configure_optimizers(self):
        """ configure_optimizers for LatentDiffusion + lora """
        ## update lora parameters
        if self.inject_unet:
            params = list(filter(lambda p: p.requires_grad, self.model.diffusion_model.parameters()))
        if self.inject_clip:
            if self.inject_unet:
                params = params + list(itertools.chain(*self.lora_require_grad_params_clip))
            else:
                params = list(itertools.chain(*self.lora_require_grad_params_clip))
        params+= list(filter(lambda p: p.requires_grad, self.mapper.parameters()))
        
        if self.learn_logvar:
            mainlogger.info('Diffusion model optimizing logvar')
            if isinstance(params[0], dict):
                params.append({"params": [self.logvar]})
            else:
                params.append(self.logvar)
        
        if self.update_keys is not None:
            params_trained = []
            for n, p in self.model.named_parameters():
                print(f'n={n}')
                if n in self.update_keys:
                    print(f"Also update parameter: {n}")
                    # params_trained.append(p)
                    params.append(p)
        ## optimizer
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
        
        ## lr scheduler
        if self.use_scheduler:
            mainlogger.info("Setting up LambdaLR scheduler...")
            lr_scheduler = self.configure_schedulers(optimizer)
            return [optimizer], [lr_scheduler]
        
        return optimizer
    
    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False, is_imgbatch=False):
        ## image/video shape: b, c, t, h, w
        data_key = 'jpg' if is_imgbatch else self.first_stage_key
        x = super().get_input(batch, data_key)
        if is_imgbatch:
            ## pack image as video
            #x = x[:,:,None,:,:]
            b = x.shape[0] // self.temporal_length
            x = rearrange(x, '(b t) c h w -> b c t h w', b=b, t=self.temporal_length)
        x_ori = x
        ## encode video frames x to z via a 2D encoder
        z = self.encode_first_stage(x)
                
        ## get caption condition
        cond_key = 'txt' if is_imgbatch else self.cond_stage_key
        cond = batch[cond_key]
        image_feature = batch['clip_img_feature']
        image_feature = self.mapper(image_feature)
        if random_uncond and self.uncond_type == 'empty_seq':
            for i, ci in enumerate(cond):
                if random.random() < self.uncond_prob:
                    cond[i] = ""
        if isinstance(cond, dict) or isinstance(cond, list):
            cond_emb = self.get_learned_conditioning(cond, image_feature=image_feature)
        else:
            cond_emb = self.get_learned_conditioning(cond.to(self.device), image_feature=image_feature.to(self.device))
        if random_uncond and self.uncond_type == 'zero_embed':
            for i, ci in enumerate(cond):
                if random.random() < self.uncond_prob:
                    cond_emb[i] = torch.zeros_like(ci)
        
        out = [z, cond_emb]
        ## optional output: self-reconst or caption
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x_ori, xrec])
        if return_original_cond:
            out.append(cond)

        return out

    # TODO: add mapper to forward
    def get_learned_conditioning(self, c, image_feature=None):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c, input_ids = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c, input_ids = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        inj_index = []
        for input_id in input_ids:
            indexs=[]
            for idx, id in enumerate(input_id):
                if id == self.cond_stage_model.modifier_token_id[0]:
                    indexs.append(idx)
            inj_index.append(indexs)
        if image_feature is not None:
            new_inputs_embeds = c.clone()
            emb_length = image_feature.shape[1]
            for bsz, idx_list in enumerate(inj_index):
                if len(idx_list)==0:
                    break
                start_idx_list = torch.where(input_ids[bsz] == idx_list[0])[0]
                for start_idx in start_idx_list:
                    end_idx = start_idx + len(idx_list) - 1
                    if len(idx_list) > emb_length:
                        lll = new_inputs_embeds[bsz, end_idx + 1:].shape[0]
                        try:
                            new_inputs_embeds[bsz, start_idx+emb_length:] = torch.cat([c[bsz, end_idx+1:end_idx+1+lll], c[bsz, -(len(idx_list) - emb_length):]], dim=0)
                        except:
                            print(f'Index Error: point1, {start_idx}, {end_idx}, {new_inputs_embeds[bsz, start_idx+emb_length:].size()}, {c[bsz, end_idx+1:end_idx+1+lll].size()}, {c[bsz, -(len(idx_list) - emb_length):].size()}')
                    else:
                        lll = new_inputs_embeds[bsz, start_idx+emb_length:].shape[0]
                        try:
                            new_inputs_embeds[bsz, start_idx+emb_length:] = c[bsz, end_idx+1:end_idx+1+lll]
                        except:
                            print(f'Index Error: point2, {start_idx}, {end_idx}, {new_inputs_embeds[bsz, start_idx+emb_length:].size()}, {c[bsz, end_idx+1:end_idx+1+lll].size()}')
                    try:
                        new_inputs_embeds[bsz, start_idx:start_idx+emb_length] = image_feature[bsz]
                    except:
                        remain_length = new_inputs_embeds[bsz, start_idx:start_idx+emb_length].size(0)
                        new_inputs_embeds[bsz, start_idx:start_idx+emb_length] = image_feature[bsz, :remain_length]
            c = new_inputs_embeds
        return c