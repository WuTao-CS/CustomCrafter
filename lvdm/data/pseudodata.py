import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PseudoData(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 is_video=True,
                 ):
        self.size = size
        self.is_video=is_video
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return 1

    def __getitem__(self, i):
        example = {}
        image = Image.open('/home/v-yanhwang/code/videostablediffusion/outputs/txt2img-samples/samples/00000.png')
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size))

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["jpg"] = (image / 127.5 - 1.0).astype(np.float32)
        if self.is_video:
            pass
        example["txt"] = 'a professional photograph of an [v] astronaut riding a horse'
        return example


class PseudoDataTrain(PseudoData):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/church_outdoor_train.txt", data_root="data/lsun/churches", **kwargs)


class PseudoDataVal(PseudoData):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/lsun/church_outdoor_val.txt", data_root="data/lsun/churches",
                         flip_p=flip_p, **kwargs)


# class LSUNBedroomsTrain(LSUNBase):
#     def __init__(self, **kwargs):
#         super().__init__(txt_file="data/lsun/bedrooms_train.txt", data_root="data/lsun/bedrooms", **kwargs)


# class LSUNBedroomsValidation(LSUNBase):
#     def __init__(self, flip_p=0.0, **kwargs):
#         super().__init__(txt_file="data/lsun/bedrooms_val.txt", data_root="data/lsun/bedrooms",
#                          flip_p=flip_p, **kwargs)


# class LSUNCatsTrain(LSUNBase):
#     def __init__(self, **kwargs):
#         super().__init__(txt_file="data/lsun/cat_train.txt", data_root="data/lsun/cats", **kwargs)


# class LSUNCatsValidation(LSUNBase):
#     def __init__(self, flip_p=0., **kwargs):
#         super().__init__(txt_file="data/lsun/cat_val.txt", data_root="data/lsun/cats",
#                          flip_p=flip_p, **kwargs)
