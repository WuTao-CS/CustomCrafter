import logging
mainlogger = logging.getLogger('mainlogger')

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


from utils.utils import instantiate_from_config

from lvdm.common import default

from lvdm.models.ddpm3d import LatentDiffusion

    
class DreamVideo(LatentDiffusion):
    def __init__(self,
                text_model_type="openclip",
                cond_stage_trainable=False,
                add_token=False,
                only_first_frame=False,
                *args, **kwargs):
        self.add_token = add_token
        self.cond_stage_trainable = cond_stage_trainable
        self.text_model_type = text_model_type
        self.only_first_frame = only_first_frame
        super().__init__(cond_stage_trainable=cond_stage_trainable, *args, **kwargs)
        self._freeze_model()
        
    def _freeze_model(self):
        for x in self.model.diffusion_model.named_parameters():
            if 'transformer_blocks' in x[0] and 'init_attn' not in x[0] and 'adapter' in x[0]:
                x[1].requires_grad = True
            else:
                x[1].requires_grad = False
                    
    def get_learnable_params(self):
        """ get learnable params """
        params = []
        for name, para in self.named_parameters():
            if para.requires_grad:
                params.append(name)
        return params
        
    def configure_optimizers(self):
        lr = self.learning_rate
        
        params = list(filter(lambda p: p.requires_grad, self.model.diffusion_model.parameters()))

        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            if self.add_token:
                if self.text_model_type == "clip":
                    params = params + list(self.cond_stage_model.transformer.text_model.embeddings.token_embedding.parameters())
                elif self.text_model_type == "openclip":
                    params = params + list(self.cond_stage_model.model.token_embedding.parameters()) # [torch.Size([49409, 1024])]
            else:
                params = params + list(self.cond_stage_model.parameters())
        for param in params:
            print("In Configure OpT")
            print(param.shape)
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    def p_losses(self, x_start, cond, t, noise=None, skip_qsample=False, x_noisy=None, cond_mask=None, mask=None, **kwargs,):
        if not skip_qsample:
            noise = default(noise, lambda: torch.randn_like(x_start))
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        else:
            assert(x_noisy is not None)
            assert(noise is not None)
        model_output = self.apply_model(x_noisy, t, cond, **kwargs)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()
        
        loss_simple = self.get_loss(model_output, target, mean=False)
        if self.only_first_frame:
            loss_simple = loss_simple[:, :, :1, ...]
        if mask is not None:
            loss_simple = (loss_simple*mask).sum([1, 2, 3, 4]) / mask.sum([1, 2, 3, 4])
        else:
            loss_simple = loss_simple.mean([1, 2, 3, 4])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
        if self.logvar.device != self.device:
            self.logvar = self.logvar.to(self.device)
        logvar_t = self.logvar[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False)
        if mask is not None:
            loss_vlb = (loss_vlb*mask).sum([1, 2, 3, 4]) / mask.sum([1, 2, 3, 4])
        else:
            loss_vlb = loss_vlb.mean([1, 2, 3, 4])
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})
            
        return loss, loss_dict
    @torch.no_grad()
    def get_input_withmask(self, batch, **args):
        out = super().get_batch_input(batch, self.first_stage_key, **args)
        mask = batch["mask"]
        if len(mask.shape) == 4:
            mask = mask[..., None]
        mask = mask.to(memory_format=torch.contiguous_format).float()
        out += [mask]
        return out

    def training_step(self, batch, batch_idx):
        if isinstance(batch, list):
            train_batch = batch[0]
            train2_batch = batch[1]
            loss_train, loss_dict = self.shared_step(train_batch)
            loss_train2, _ = self.shared_step(train2_batch)
            loss = loss_train + loss_train2
        else:
            train_batch = batch
            loss, loss_dict = self.shared_step(train_batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def shared_step(self, batch, **kwargs):
        x, c, mask = self.get_input_withmask(batch, **kwargs)
        loss = self(x, c, mask=mask)
        return loss
        
