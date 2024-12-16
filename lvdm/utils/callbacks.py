import logging
import time, os
import numpy as np
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf

import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from utils.save_video import log_local, prepare_to_log

mainlogger = logging.getLogger('mainlogger')
# ---------------------------------------------------------------------------------
class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images=8, clamp=True, rescale=True, save_dir=None, \
                to_local=True, log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.to_local = to_local
        self.clamp = clamp
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        if self.to_local:
            ## default save dir
            self.save_dir = os.path.join(save_dir, "images")
            os.makedirs(os.path.join(self.save_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "val"), exist_ok=True)

    def log_to_tensorboard(self, pl_module, batch_logs, filename, split, save_fps=10):
        """ log images and videos to tensorboard """        
        global_step = pl_module.global_step
        for key in batch_logs:
            value = batch_logs[key]
            tag = "gs%d-%s/%s-%s"%(global_step, split, filename, key)
            if isinstance(value, list) and isinstance(value[0], str):
                captions = ' |------| '.join(value)
                pl_module.logger.experiment.add_text(tag, captions, global_step=global_step)
            elif isinstance(value, torch.Tensor) and value.dim() == 5:
                video = value
                n = video.shape[0]
                video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
                frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n)) for framesheet in video] #[3, n*h, 1*w]
                grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
                grid = (grid + 1.0) / 2.0
                grid = grid.unsqueeze(dim=0)
                pl_module.logger.experiment.add_video(tag, grid, fps=save_fps, global_step=global_step)
            elif isinstance(value, torch.Tensor) and value.dim() == 4:
                img = value
                grid = torchvision.utils.make_grid(img, nrow=int(n))
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                pl_module.logger.experiment.add_image(tag, grid, global_step=global_step)
            else:
                pass

    @rank_zero_only
    def log_batch_imgs(self, pl_module, batch, batch_idx, split="train"):
        skip_freq = self.batch_freq if split == "train" else 5
        if (batch_idx+1) % skip_freq == 0:
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                log_func = pl_module.log_images
                batch_logs = log_func(batch, split=split, **self.log_images_kwargs)
            
            ## process: move to CPU and clamp
            batch_logs = prepare_to_log(batch_logs, self.max_images, self.clamp)
            torch.cuda.empty_cache()
            
            filename = "ep{}_idx{}_rank{}".format(
                pl_module.current_epoch,
                batch_idx,
                pl_module.global_rank)
            if self.to_local:
                mainlogger.info("Log [%s] batch <%s> to local ..."%(split, filename))
                filename = "gs{}_".format(pl_module.global_step) + filename
                log_local(batch_logs, os.path.join(self.save_dir, split), filename, save_fps=10)
            else:
                mainlogger.info("Log [%s] batch <%s> to tensorboard ..."%(split, filename))
                self.log_to_tensorboard(pl_module, batch_logs, filename, split, save_fps=10)
            mainlogger.info('Finish!')

            if is_train:
                pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if self.batch_freq != -1 and self.save_dir:
            self.log_batch_imgs(pl_module, batch, batch_idx, split="train")


# ---------------------------------------------------------------------------------
class CUDACallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        # lightning update
        if int((pl.__version__).split('.')[1])>=7:
            gpu_index = trainer.strategy.root_device.index
        else:
            gpu_index = trainer.root_gpu
        torch.cuda.reset_peak_memory_stats(gpu_index)
        torch.cuda.synchronize(gpu_index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if int((pl.__version__).split('.')[1])>=7:
            gpu_index = trainer.strategy.root_device.index
        else:
            gpu_index = trainer.root_gpu
        torch.cuda.synchronize(gpu_index)
        max_memory = torch.cuda.max_memory_allocated(gpu_index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)
            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------------
# for lower lighting
class SetupCallback_low(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config, auto_resume, save_ptl_log=False):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.auto_resume = auto_resume
        self.save_ptl_log = save_ptl_log

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last_summoning.ckpt")
            trainer.save_checkpoint(ckpt_path)

        # for old version lightning
    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)
            
            if self.save_ptl_log:
                logger = logging.getLogger("pytorch_lightning")
                logger.addHandler(logging.FileHandler(os.path.join(self.logdir, "ptl_log.log")))

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            pass

# for higher lighting
class SetupCallback_high(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config, auto_resume, save_ptl_log=False):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.auto_resume = auto_resume
        self.save_ptl_log = save_ptl_log

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last_summoning.ckpt")
            trainer.save_checkpoint(ckpt_path)

    # RuntimeError: The `Callback.on_pretrain_routine_start` hook was removed in v1.8. Please use `Callback.on_fit_start` instead.
    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)
            
            if self.save_ptl_log:
                logger = logging.getLogger("pytorch_lightning")
                logger.addHandler(logging.FileHandler(os.path.join(self.logdir, "ptl_log.log")))

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))
