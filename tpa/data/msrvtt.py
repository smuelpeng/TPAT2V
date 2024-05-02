import json
import math
import os
import random
from dataclasses import dataclass, field
import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transform
from PIL import Image
import imageio
import cv2
import pandas as pd
from collections import defaultdict

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import tpa
from ..utils.config import parse_structured
from ..utils.typing import *
from ..utils.misc import load_json
from ..utils.video_capture import VideoCapture

def init_transform_dict(input_res=224):
    tsfm_dict = {
        'clip_val': transforms.Compose([
            transforms.Resize(input_res, interpolation=Image.BICUBIC),
            transforms.CenterCrop(input_res),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]),
        'clip_train': transforms.Compose([
            transforms.RandomResizedCrop(input_res, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0, saturation=0, hue=0),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    }

    return tsfm_dict

@dataclass
class MSRVTTDataModuleConfig:
    videos_dir: str = "videos"
    split_type: str = "train"
    dir: str = "data/MSRVTT"
    db_file: str = "MSRVTT_data.json"
    train_file: str = "9k"
    train_csv: str = "MSRVTT_train.9k.csv"
    test_csv: str = "MSRVTT_JSFUSION_test.csv"
    input_res: int = 224
    eval_batch_size: int = 32
    batch_size: int = 32
    num_workers: int = 8
    num_frames: int = 12
    video_sample_type: str = "uniform"



class MSRVTTDataset(Dataset):
    # def __init__(self, config: Config, split_type = 'train', img_transforms=None):
    def __init__(self, config: Any, split_type: str = "train",
               ) -> None:
        self.config = config
        self.videos_dir = config.videos_dir
        img_transforms = init_transform_dict(config.input_res)
        self.img_transforms = img_transforms[f'clip_{split_type}']
        self.split_type = split_type

        self.db = load_json(config.db_file)
        if split_type == 'train':
            train_df = pd.read_csv(config.train_csv)
            self.train_vids = train_df['video_id'].unique()
            self._compute_vid2caption()
            self._construct_all_train_pairs()
        else:
            self.test_df = pd.read_csv(config.test_csv)

            
    def __getitem__(self, index):

        if self.split_type == 'train':
            video_path, caption, video_id, sen_id = self._get_vidpath_and_caption_by_index(index)
            imgs, idxs = VideoCapture.load_frames_from_video(video_path,
                                                             self.config.num_frames,
                                                             self.config.video_sample_type)
            if self.img_transforms is not None:
                imgs = self.img_transforms(imgs)
            return {
                'video_id': video_id,
                'video': imgs,
                'text': caption,
            }
        else:
            video_path, caption, video_id = self._get_vidpath_and_caption_by_index(index)
            imgs, idxs = VideoCapture.load_frames_from_video(video_path,
                                                             self.config.num_frames,
                                                             self.config.video_sample_type)
            if self.img_transforms is not None:
                imgs = self.img_transforms(imgs)
            return {
                'video_id': video_id,
                'video': imgs,
                'text': caption,
            }

    def __len__(self):
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        return len(self.test_df)


    def _get_vidpath_and_caption_by_index(self, index):
        if self.split_type == 'train':
            vid, caption, senid = self.all_train_pairs[index]
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
            return video_path, caption, vid, senid
        else:
            vid = self.test_df.iloc[index].video_id
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
            caption = self.test_df.iloc[index].sentence
            return video_path, caption, vid

    
    def _construct_all_train_pairs(self):
        self.all_train_pairs = []
        if self.split_type == 'train':
            for vid in self.train_vids:
                for caption, senid in zip(self.vid2caption[vid], self.vid2senid[vid]):
                    self.all_train_pairs.append([vid, caption, senid])
            
    def _compute_vid2caption(self):
        self.vid2caption = defaultdict(list)
        self.vid2senid   = defaultdict(list)

        for annotation in self.db['sentences']:
            caption = annotation['caption']
            vid = annotation['video_id']
            self.vid2caption[vid].append(caption)
            senid = annotation['sen_id']
            self.vid2senid[vid].append(senid)

class MSRVTTDataModule(pl.LightningDataModule):
    cfg: MSRVTTDataModuleConfig
    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(MSRVTTDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = MSRVTTDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = MSRVTTDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = MSRVTTDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            # collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            # collate_fn=self.val_dataset.collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            # collate_fn=self.test_dataset.collate,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

