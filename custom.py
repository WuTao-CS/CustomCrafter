import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch

import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
from lvdm.data.base import Txt2ImgIterableBaseDataset
from utils.utils import instantiate_from_config

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

def print_learnable_params(model):
    # 获取优化器
    optimizer = model.configure_optimizers()
    cnt = 0
    print("Learnable parameters: ")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            cnt+=1
    print("Total number of learnable parameters: ", cnt)
    # 如果有多个优化器，你可能需要遍历它们
    if isinstance(optimizer, list):
        for idx, opt in enumerate(optimizer):
            print(f"Optimizer {idx}:")
            for param_group in opt.param_groups:
                for name, param in model.lightning_module.named_parameters():
                    if param.requires_grad:
                        print(f"Parameter: {name}, Size: {param.size()}")
    else:
        cnt_opt = 0
        for param_group in optimizer.param_groups:
            for item in param_group['params']:
                # print(item.size())
                cnt_opt+=1
        print("Total number of learnable parameters in optimizer: ", cnt_opt)
            # print(param_group['params'][0].size())
    return
    
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    config.model.params.ckpt_path = ckpt
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    del pl_sd
    del sd
    return model
def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "--with_prior_preservation",
        type=str2bool,
        default=False,
        nargs="?",
        help="with_prior_preservation",
    )
    parser.add_argument(
        "--lora",
        type=str2bool,
        default=False,
        nargs="?",
        help="lora",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--base_learning_rate",
        type=float,
        default=None,
        help="base_learning_rate",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument("--auto_resume", type=str2bool, nargs="?", const=False, default=False, help="")
    parser.add_argument(
        "--datapath",
        type=str,
        default="",
        help="path to target images",
    )
    parser.add_argument(
        "--datamaskpath",
        type=str,
        default=None,
        help="path to datamaskpath images",
    )
    parser.add_argument(
        "--reg_datapath",
        type=str,
        default=None,
        help="path to regularization images",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="",
        help="path to target images",
    )
    parser.add_argument(
        "--reg_caption",
        type=str,
        default="",
        help="path to target images",
    ) 
    parser.add_argument(
        "--datapath2",
        type=str,
        default=None,
        help="path to target images",
    )
    parser.add_argument(
        "--reg_datapath2",
        type=str,
        default=None,
        help="path to regularization images",
    )
    parser.add_argument(
        "--caption2",
        type=str,
        default=None,
        help="path to target images",
    )
    parser.add_argument(
        "--reg_caption2",
        type=str,
        default=None,
        help="path to regularization images' caption",
    )
    parser.add_argument(
        "--modifier_token",
        type=str,
        default=None,
        help="token added before cateogry word for personalization use case",
    )
    parser.add_argument(
        "--initializer_token",
        type=str,
        default=None,
        help="cateogry word for personalization use case",
    )
    parser.add_argument(
        "--modifier_token_init_weight",
        type=str,
        default=None,
        help="modifier token init weight path for personalization use case",
    )
    parser.add_argument(
        "--freeze_model",
        type=str,
        default="crossattn",
        help="crossattn to enable fine-tuning of all key, value, query matrices",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="repeat the target dataset by how many times. Used when training without regularization",
    ) 
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="overwrite batch size",
    )
    parser.add_argument(
        "-rc",
        "--resume-from-checkpoint-custom",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    return parser

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)

def collate_fn(examples, with_prior_preservation):
    input_ids = [example["prompt"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    mask = [example["mask"] for example in examples]
    pixel_values = torch.stack(pixel_values)
    mask = torch.stack(mask)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    mask = mask.to(memory_format=torch.contiguous_format).float()
    instance_batch = {"caption": input_ids, "video": pixel_values, "mask": mask}
    batch = instance_batch

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        reg_input_ids = [example["class_prompt"] for example in examples]
        reg_pixel_values = [example["class_images"] for example in examples]
        reg_mask = [example["class_mask"] for example in examples]
        reg_pixel_values = torch.stack(reg_pixel_values)
        reg_mask = torch.stack(reg_mask)
        reg_pixel_values = reg_pixel_values.to(memory_format=torch.contiguous_format).float()
        reg_mask = reg_mask.to(memory_format=torch.contiguous_format).float()
        reg_batch = {"caption": reg_input_ids, "video": reg_pixel_values, "mask": reg_mask}
        batch = [instance_batch, reg_batch]

    # input_ids = torch.stack(input_ids)
    # pixel_values = torch.stack(pixel_values)
    # mask = torch.stack(mask)
    # pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    # mask = mask.to(memory_format=torch.contiguous_format).float()
    return batch

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, train2=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False, with_prior_preservation=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        self.with_prior_preservation = with_prior_preservation
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if train2 is not None and train2['params']['caption'] != '':
            self.dataset_configs["train2"] = train2
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        if "train2" in self.dataset_configs and self.dataset_configs["train2"]['params']["caption"] != '':
            train_set = self.datasets["train"]
            train2_set = self.datasets["train2"]
            concat_dataset = ConcatDataset(train_set, train2_set)
            if self.with_prior_preservation:
                return DataLoader(concat_dataset, batch_size=self.batch_size // 2,
                                num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                                worker_init_fn=init_fn, collate_fn=lambda examples: collate_fn(examples, self.with_prior_preservation),)
            else:
                return DataLoader(concat_dataset, batch_size=self.batch_size,
                                num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                                worker_init_fn=init_fn, collate_fn=lambda examples: collate_fn(examples, self.with_prior_preservation))
        else:
            if self.with_prior_preservation:
                return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                                num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                                worker_init_fn=init_fn, collate_fn=lambda examples: collate_fn(examples, self.with_prior_preservation),)
            else:
                return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                                num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                                worker_init_fn=init_fn,collate_fn=lambda examples: collate_fn(examples, self.with_prior_preservation),)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

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
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""

        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        if "accelerator" not in trainer_config:
            # lightining update
            if int((pl.__version__).split('.')[1])>=7:
                trainer_config["accelerator"] = "cuda"
            else:
                trainer_config["accelerator"] = "ddp"
            print('Set DDP mode')

        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        
        if not "gpus" in trainer_config:
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False

        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        #model
        config.data.params.train.params.caption = opt.caption
        config.data.params.train.params.reg_caption = opt.reg_caption
        config.data.params.train.params.datapath = opt.datapath
        config.data.params.train.params.reg_datapath = opt.reg_datapath
        if opt.datamaskpath is not None:
            config.data.params.train.params.datamaskpath = opt.datamaskpath

        if opt.caption2 is not None:
            config.data.params.train2.params.caption = opt.caption2
            config.data.params.train2.params.reg_caption = opt.reg_caption2
            config.data.params.train2.params.datapath = opt.datapath2
            config.data.params.train2.params.reg_datapath = opt.reg_datapath2
        config.data.params.validation = config.data.params.train
        if opt.batch_size is not None:
            config.data.params.batch_size = opt.batch_size
        if opt.base_learning_rate is not None:
            config.model.base_learning_rate = opt.base_learning_rate
        if opt.modifier_token is not None:
            config.model.params.cond_stage_config.params.modifier_token = opt.modifier_token 
        if opt.initializer_token is not None:
            config.model.params.cond_stage_config.params.initializer_token = opt.initializer_token 
        if opt.repeat > 0:
            config.data.params.train.params.repeat = opt.repeat
        if opt.resume_from_checkpoint_custom:
            config.model.params.ckpt_path = None
        if opt.freeze_model is not None:
            config.model.params.freeze_model = opt.freeze_model
        
        model = instantiate_from_config(config.model)

        if opt.resume_from_checkpoint_custom:
            print("Resume from:",opt.resume_from_checkpoint_custom)
            st = torch.load(opt.resume_from_checkpoint_custom, map_location='cpu')["state_dict"]
            token_weights = st["cond_stage_model.model.token_embedding.weight"]
            del st["cond_stage_model.model.token_embedding.weight"]
            missing_key,unexpected_keys = model.load_state_dict(st, strict=False)
            print("Missing key: ",missing_key)
            print("Unexpected key: ",unexpected_keys)
            if opt.lora:
                model._inject_lora()
            model.cond_stage_model.model.token_embedding.weight.data[:token_weights.shape[0]] = token_weights.data
            if opt.modifier_token_init_weight is not None:
                print("Loading modifier token init weight from: ", opt.modifier_token_init_weight)
                init_token_weights = torch.load(opt.modifier_token_init_weight)['<new1>']
                for modifier_id in model.cond_stage_model.modifier_token_id:
                    model.cond_stage_model.model.token_embedding.weight.data[modifier_id] = init_token_weights
                del init_token_weights
            else:
                for modifier_id, initializer_id in zip(model.cond_stage_model.modifier_token_id, model.cond_stage_model.initializer_token_id):
                    model.cond_stage_model.model.token_embedding.weight.data[modifier_id] = token_weights[initializer_id]
            del st
        # trainer and callbacks
        trainer_kwargs = dict()
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                # https://github.com/Lightning-AI/lightning/issues/13958
                # The test-tube package is no longer maintained and PyTorch Lightning will remove the :class:´TestTubeLogger´ in v1.7.0.
                "target": "pytorch_lightning.loggers.CSVLogger" if int((pl.__version__).split('.')[1])>=7 else "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
                "every_n_epochs": 20,
                "save_top_k": 50,
                "monitor":"epoch"
            }
        }
        # if hasattr(model, "monitor"):
        #     print(f"Monitoring {model.monitor} as checkpoint metric.")
        #     default_modelckpt_cfg["params"]["monitor"] = model.monitor
        #     default_modelckpt_cfg["params"]["save_top_k"] = -1
        #     # default_modelckpt_cfg["params"]["every_n_epochs"] = 1
        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "lvdm.utils.callbacks.SetupCallback_high" if int((pl.__version__).split('.')[1])>=7 else "lvdm.utils.callbacks.SetupCallback_low" ,
                "params": {
                    "resume": '',
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                    "auto_resume": opt.auto_resume,
                }
            },
            "batch_logger": {
                "target": "lvdm.utils.callbacks.ImageLogger",
                "params": {
                    "save_dir": logdir,
                    "batch_frequency": 200,
                    "max_images": 10,
                    "to_local": True,
                    "clamp": True,
                    "log_images_kwargs": {
                        "ddim_steps": 50,
                        "unconditional_guidance_scale": 15.0
                    }
                }
            },
            "learning_rate_logger": {
                "target": "lvdm.utils.callbacks.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                }
            },
            "cuda_callback": {
                "target": "lvdm.utils.callbacks.CUDACallback"
            },
        }
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                         "filename": "{epoch:06}-{step:09}",
                         "verbose": True,
                         'save_top_k': -1,
                         'every_n_train_steps': 50,
                         'save_weights_only': True
                     }
                     }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        # default strategy config
        default_strategy_dict = {
            "target": "pytorch_lightning.strategies.DDPShardedStrategy"
        }

        if "strategy" in lightning_config:
            strategy_cfg = lightning_config.strategy
        else:
            strategy_cfg = OmegaConf.create()
            strategy_cfg = OmegaConf.merge(default_strategy_dict, strategy_cfg)

        if int((pl.__version__).split('.')[1])>=7:
            trainer_kwargs['precision'] = lightning_config.get('precision', 32)
            print(f'set precision={trainer_kwargs["precision"]}')
            print('lightning_config',lightning_config)
            # strategy can be str
            if type(strategy_cfg) == str:
                trainer_kwargs["strategy"] = strategy_cfg
            else:
                # default strategy is ddp shared
                trainer_kwargs["strategy"] = instantiate_from_config(strategy_cfg)
            print(f'strategy')
            print(trainer_kwargs["strategy"])
        else:
            print('low version ptl, no ddp shared')
            find_unused_parameters=lightning_config.get("find_unused_parameters", False)
            trainer_kwargs["plugins"] = DDPPlugin(find_unused_parameters=find_unused_parameters)

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir

        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1

        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
            # adjust the log batch freq to the actual forward steps (not the optimize step)
            lightning_config.callbacks.image_logger.params.batch_frequency = lightning_config.callbacks.image_logger.params.batch_frequency / accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches

        # data
        if getattr(config.data, 'auto_cal_bs', False):
            bs_per_gpu = config.data.params.batch_size * accumulate_grad_batches
            total_bs = ngpu * lightning_config.trainer.num_nodes \
                       * bs_per_gpu
            print(f'Actual total batch size = {total_bs}')
            config.data.params.train.params['bs_per_gpu'] = bs_per_gpu
            if "validation" in config.data.params:
                config.data.params.validation.params['bs_per_gpu'] = bs_per_gpu

        data = instantiate_from_config(config.data)
        data.prepare_data()
        data.setup()
        print("#### Data #####")
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if opt.with_prior_preservation:
            base_lr = base_lr * 2
        scale_lr = opt.scale_lr and getattr(config.model, 'scale_lr', True)
        if scale_lr:
            num_nodes = 1
            model.learning_rate = ngpu * num_nodes * bs * base_lr * accumulate_grad_batches
            print("Setting learning rate to {:.2e} = {} (num_gpus) * {} (num_nodes) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, ngpu, num_nodes, bs, base_lr))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last_summoning.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()

        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            print_learnable_params(model)
            trainer.fit(model, data)
            # try:
            #     trainer.fit(model, data)
            # except Exception:
            #     melk()
            #     raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())
