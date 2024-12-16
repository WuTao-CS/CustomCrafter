import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import random
import PIL
from pathlib import Path
from utils.save_video import save_video_tensor_to_mp4

templates_small = [
    'video of a {}',
    '{}',
    'a static video of a {}',
    "a {}",
]


templates_small_style = [
    'painting in the style of {}',
    'video in the style of {}',
    'movie in the style of {}',
    'image in the style of {}',
    'style of {}',
    '{} style',
    '{} art style',
    'in {} style',
    'with {} style',
    'in the style of {} art',
    'in the style of {}',
]


def isimage(path):
    if 'png' in path.lower() or 'jpg' in path.lower() or 'jpeg' in path.lower():
        return True

class MaskBase(Dataset):
    def __init__(self,
                 datapath,
                 num_frames=16,
                 first_stage_key="video",
                 cond_stage_key="caption",
                 reg_datapath=None,
                 caption=None,
                 reg_caption=None,
                 size=[320,512],
                 interpolation="bicubic",
                 flip_p=0.5,
                 aug=False,
                 style=False,
                 with_prior_preservation=False,
                 ):

        self.aug = aug
        self.style = style
        self.templates_small = templates_small
        if self.style:
            self.templates_small = templates_small_style
        if os.path.isdir(datapath):
            instanse_images_path = [os.path.join(datapath, file_path) for file_path in os.listdir(datapath) if isimage(file_path)]
        else:
            with open(datapath, "r") as f:
                instanse_images_path = f.read().splitlines()
        self.with_prior_preservation = with_prior_preservation
        self.instanse_image_length = len(instanse_images_path)
        reg_images_path = []
        self.reg_image_length = 0
        if reg_datapath is not None:
            if os.path.isdir(reg_datapath):
                reg_images_path = [os.path.join(reg_datapath, file_path) for file_path in os.listdir(reg_datapath) if isimage(file_path)]
            else:
                with open(reg_datapath, "r") as f:
                    reg_images_path = f.read().splitlines()
            self.reg_image_length = len(reg_images_path)

        self.labels = {
            "relative_file_path1_": [x for x in instanse_images_path],
            "relative_file_path2_": [x for x in reg_images_path],
        }

        self.size = size
        self.interpolation = {"bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.caption = caption

        if os.path.exists(self.caption):
            self.caption = [x.strip() for x in open(caption, 'r').readlines()]

        self.reg_caption = reg_caption
        if os.path.exists(self.reg_caption):
            self.reg_caption = [x.strip() for x in open(reg_caption, 'r').readlines()]
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.num_frames = num_frames

    def __len__(self):
        if self.reg_image_length > 0:
            # return 10
            return self.reg_image_length
        else:
            return self.instanse_image_length

    def image2video(self, input_image, mask):
        input_image = np.transpose(input_image, (2, 0, 1))
        input_image = np.expand_dims(input_image, axis=1)
        input_image = np.repeat(input_image, repeats=self.num_frames, axis=1)
        mask = np.expand_dims(mask, axis=0)
        mask = np.repeat(mask, repeats=self.num_frames, axis=0)
        mask = np.expand_dims(mask, axis=0)
        return input_image, mask

    def augment(self, image, prompt):
        width, height = image.size

        # 计算16:10的宽高比
        aspect_ratio = self.size[1] / self.size[0]

        # 计算新的宽度和高度
        new_width = width
        new_height = int(new_width / aspect_ratio)

        # 如果新的高度大于原始高度，则需要调整宽度
        if new_height > height:
            new_height = height
            new_width = int(new_height * aspect_ratio)

        # 计算裁剪区域的左上角和右下角坐标
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2

        # 裁剪图片
        image = image.crop((left, top, right, bottom))
        assert new_height < new_width
        image = image.resize((self.size[1], self.size[0]), resample=self.interpolation)

        if np.random.randint(0, 3) < 2:
            random_scale_factor = np.random.randint(3, 11) / 10
        else:
            random_scale_factor = np.random.randint(12, 15) / 10
        random_scale_height = int(random_scale_factor * self.size[0])
        if random_scale_height % 2 == 1:
            random_scale_height += 1
        random_scale_width = int(random_scale_height * aspect_ratio)
        if random_scale_width % 2 == 1:
            random_scale_width += 1
        if random_scale_height < 0.6 * self.size[0]:
            add_to_caption = np.random.choice(["a far away ", "very small "])
            prompt = add_to_caption + prompt
            x = np.random.randint(0, self.size[1] - random_scale_width + 1)
            y = np.random.randint(0, self.size[0] - random_scale_height + 1)

            image = image.resize((random_scale_width, random_scale_height), resample=self.interpolation)
            image = np.array(image).astype(np.uint8)
            # image = ((image / 255 - 0.5) * 2).astype(np.float32) 

            input_image = np.zeros((self.size[0], self.size[1], 3), dtype=np.float32)
            input_image[y:y+random_scale_height, x:x+random_scale_width, :] = image
            input_image = ((input_image / 255 - 0.5) * 2).astype(np.float32) 

            mask = np.zeros((self.size[0] // 8, self.size[1] // 8))
            mask[y//8:(y+random_scale_height)//8, x//8:(x+random_scale_width)//8] = 1.

        elif random_scale_height > self.size[0]:
            add_to_caption = np.random.choice(["zoomed in ", "close up "])
            prompt = add_to_caption + prompt

            x = np.random.randint(self.size[1] // 2, random_scale_width - self.size[1] // 2 + 1)
            y = np.random.randint(self.size[0] // 2, random_scale_height - self.size[0] // 2 + 1)

            image = image.resize((random_scale_width, random_scale_height), resample=self.interpolation)
            image = np.array(image).astype(np.uint8)
            # image = (image / 127.5 - 1.0).astype(np.float32)
            input_image = image[y - self.size[0] // 2: y + self.size[0] // 2, x - self.size[1] // 2: x + self.size[1] // 2,  :]
            input_image = ((input_image / 255 - 0.5) * 2).astype(np.float32) 
            mask = np.ones((self.size[0] // 8, self.size[1] // 8))
        else:
            if self.size is not None:
                image = image.resize((self.size[1], self.size[0]), resample=self.interpolation)
            input_image = np.array(image).astype(np.uint8)
            # input_image = (input_image / 127.5 - 1.0).astype(np.float32)
            input_image = ((input_image / 255 - 0.5) * 2).astype(np.float32) 
            mask = np.ones((self.size[0] // 8, self.size[1] // 8))

        return input_image, mask, prompt

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.labels["relative_file_path1_"][index % self.instanse_image_length])
        if isinstance(self.caption, str):
            instance_prompt = np.random.choice(self.templates_small).format(self.caption)
        else:
            instance_prompt = self.caption[index % min(self.instanse_image_length, len(self.caption))]
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        
        instance_image = self.flip(instance_image)
        if self.aug:
            instance_image, instance_mask, instance_prompt = self.augment(instance_image, instance_prompt)
        else:
            if self.size is not None:
                instance_image = instance_image.resize((self.size[1], self.size[0]), resample=self.interpolation)
                instance_image = np.array(instance_image).astype(np.uint8)
                instance_image = ((instance_image / 255 - 0.5) * 2).astype(np.float32) 
                instance_mask = np.ones((self.size[0] // 8, self.size[1] // 8))
        instance_image, instance_mask = self.image2video(instance_image,instance_mask)
        example["instance_images"] = torch.tensor(instance_image)
        # save_video_tensor_to_mp4(example["instance_images"].unsqueeze(0), "instance.mp4", self.num_frames)
        example["mask"] = torch.tensor(instance_mask)
        example['prompt'] = instance_prompt


        if self.with_prior_preservation:
            reg_image = Image.open(self.labels["relative_file_path2_"][index % self.reg_image_length])
            if isinstance(self.reg_caption, str):
                example["class_prompt"] = np.random.choice(self.templates_small).format(self.reg_caption)
            else:
                example["class_prompt"] = self.reg_caption[index % self.reg_image_length]
            if not reg_image.mode == "RGB":
                reg_image = reg_image.convert("RGB")
            if self.size is not None:
                reg_image = reg_image.resize((self.size[1], self.size[0]), resample=self.interpolation)
            reg_image = np.array(reg_image).astype(np.uint8)
            # input_image = (input_image / 127.5 - 1.0).astype(np.float32)
            reg_image = ((reg_image / 255 - 0.5) * 2).astype(np.float32) 
            reg_mask = np.ones((self.size[0] // 8, self.size[1] // 8))
            reg_image, reg_mask = self.image2video(reg_image, reg_mask)
            example["class_images"] = torch.tensor(reg_image)
            # save_video_tensor_to_mp4(example["class_images"].unsqueeze(0), "reg_video.mp4", self.num_frames)
            example["class_mask"] = torch.tensor(reg_mask)
        return example
