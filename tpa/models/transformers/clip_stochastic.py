import torch.nn as nn
from dataclasses import dataclass, field


# from config.base_config import Config
# from modules.transformer import Transformer
# from modules.stochastic_module import StochasticText
import tpa
from ...utils.base import BaseModule
from ...utils.typing import *
from .transformers import Transformer
from ..stochastic_module import StochasticText

class CLIPStochastic(BaseModule):

    @dataclass
    class Config(BaseModule.Config):    
        clip_arch: str = 'ViT-B/32'
        embed_dim: int = 512
        num_mha_heads: int = 1
        transformer_dropout: float = 0.3
        num_frames: int = 12
        stochastic_prior: str = 'uniform01'
        stochastic_prior_std: float = 1.0        
        input_res: int = 224
        
    cfg: Config

    def configure(self) -> None:
        super().configure()
        from transformers import CLIPModel
        if self.cfg.clip_arch == 'ViT-B/32':
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        elif self.cfg.clip_arch == 'ViT-B/16':
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        else:
            raise ValueError
        
        self.pool_frames = Transformer(self.cfg)
        self.stochastic = StochasticText(self.cfg)

    def forward(self, data, return_all_frames=False, is_train=True):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        video_data = video_data.reshape(-1, 3, self.cfg.input_res, self.cfg.input_res)

        if is_train:

            text_features = self.clip.get_text_features(**text_data)
            video_features = self.clip.get_image_features(video_data)



            video_features = video_features.reshape(batch_size, self.cfg.num_frames, -1) # [bs, #F, 512]

            video_features_pooled = self.pool_frames(text_features, video_features)

            # @WJM: perform stochastic text
            text_features_stochstic, text_mean, log_var = self.stochastic(text_features, video_features)


            if return_all_frames:
                return text_features, video_features, video_features_pooled, text_features_stochstic, text_mean, log_var

            return text_features, video_features_pooled,  text_features_stochstic, text_mean, log_var

        else:
            text_features = self.clip.get_text_features(**text_data)
            video_features = self.clip.get_image_features(video_data)

            video_features = video_features.reshape(batch_size, self.cfg.num_frames, -1)
            video_features_pooled = self.pool_frames(text_features, video_features)

            # @WJM: re-parameterization for text (independent of the text-cond pooling)
            text_features_stochstic, _, _ = self.stochastic(text_features, video_features)

            if return_all_frames:
                return text_features, video_features, video_features_pooled, text_features_stochstic

            return text_features, video_features_pooled, text_features_stochstic
