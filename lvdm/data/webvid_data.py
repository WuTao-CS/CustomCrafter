import glob
import json
import os
import random
import pandas as pd
import torch
from decord import VideoReader, cpu
from einops import rearrange
import torchvision.transforms as T


class WebVidDataset(torch.utils.data.Dataset):
    """
    Taichi Dataset.
    Assumes data is structured as follows.
    Taichi/
        train/
            xxx.mp4
            ...
        test/
            xxx.mp4
            ...
    """

    def __init__(
        self,
        dataset_name,
        data_dir,
        resolution,
        video_length,
        split='train',
        cut = "10M",
        frame_stride=4,
        metadata_dir=None,
        metadata_folder_name=None, #"webvid10m_meta",
        spatial_transform="resize",
        load_method="decord",
        first_stage_key="video",
        cond_stage_key="caption",
    ):
        self.data_dir = data_dir
        self.cut = cut
        self.resolution = resolution
        self.video_length = video_length
        self.frame_stride = frame_stride
        self.spatial_transform = spatial_transform
        self.load_method = load_method
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.split = split
        self.metadata_folder_name = metadata_folder_name

        assert self.load_method in ["decord", "readvideo", "videoclips"]
        self.exts = ["avi", "mp4", "webm"]
        if metadata_dir is not None:
            self.metadata_dir = os.path.expandvars(metadata_dir)
        else:
            self.metadata_dir = self.data_dir
        if isinstance(self.resolution, int):
            self.resolution = [self.resolution, self.resolution]
        self.resolution = list(self.resolution)
        assert isinstance(self.resolution, list) and len(self.resolution) == 2
        self.max_resolution = max(self.resolution)
        if self.spatial_transform == "resize":
            print("Spatial transform: resize with no crop")
            self.video_transform = T.Resize(self.resolution)
        elif self.spatial_transform == "":
            self.video_transform = None
        else:
            raise NotImplementedError
        self._load_metadata()

    def get_data_decord(self, index):
        while True:
            item = index % len(self.metadata)
            sample = self.metadata.iloc[item]
            video_path, rel_fp = self._get_video_path(sample)
            caption = self._get_caption(sample)
            try:
                video_reader = VideoReader(
                    video_path,
                    ctx=cpu(0),
                    width=self.max_resolution,
                    height=self.max_resolution,
                )
                if len(video_reader) < self.video_length:
                    index += 1
                    continue
                else:
                    break
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
        all_frames = list(range(0, len(video_reader), self.frame_stride))
        if len(all_frames) < self.video_length:
            all_frames = list(range(0, len(video_reader), 1))
        rand_idx = random.randint(0, len(all_frames) - self.video_length)
        frame_indices = list(range(rand_idx, rand_idx + self.video_length))
        frames = video_reader.get_batch(frame_indices)
        assert frames.shape[0] == self.video_length, f"{len(frames)}, self.video_length={self.video_length}"
        frames = torch.tensor(data=frames.asnumpy()).float().permute(0, 3, 1, 2) 
        if self.video_transform is not None: 
            frames = self.video_transform(frames) 
        frames = frames.permute(1, 0, 2, 3).float() 
        assert ( frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[1] ), f"frames={frames.shape}, self.resolution={self.resolution}" 
        frames = (frames / 255 - 0.5) * 2 
        data = {self.first_stage_key: frames, self.cond_stage_key: caption}
        return data

    def get_data_readvideo(self, index):
        return

    def __getitem__(self, index):
        if self.load_method == "decord":
            data = self.get_data_decord(index)
        elif self.load_method == "readvideo":
            data = self.get_data_readvideo(index)
        return data

    def __len__(self):
        return len(self.metadata)

    def _load_metadata(self):
        assert self.metadata_folder_name is not None
        assert self.cut is not None
        metadata_dir = os.path.join(self.metadata_dir, self.metadata_folder_name) #'metadata'
        metadata_fp = os.path.join(metadata_dir, f'results_{self.cut}_{self.split}.csv')
        metadata = pd.read_csv(metadata_fp)

        if self.split == 'val':
            metadata = metadata.sample(1000, random_state=0)  # 15k val is unnecessarily large, downsample.

        metadata['caption'] = metadata['name']
        del metadata['name']
        self.metadata = metadata
        # TODO: clean final csv so this isn't necessary
        self.metadata.dropna(inplace=True)
        # self.metadata['caption'] = self.metadata['caption'].str[:350]

    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        if self.split == 'train':
            full_video_fp = os.path.join(self.data_dir, 'train_10M', rel_video_fp)
        elif self.split == 'val':
            full_video_fp = os.path.join(self.data_dir, 'val_10M', rel_video_fp)
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        return sample['caption']
