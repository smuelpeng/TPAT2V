import torch.nn as nn
from dataclasses import dataclass, field
from ...utils.base import BaseModule
import tpa


class CLIPTOKENIZER(BaseModule):

    @dataclass
    class Config(BaseModule.Config):    
        clip_arch: str = 'openai/clip-vit-base-patch32'
        
    cfg: Config

    def configure(self) -> None:
        super().configure()
        from transformers import CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(self.cfg.clip_arch, TOKENIZERS_PARALLELISM=False)

    def forward(self, data, return_all_frames=False, is_train=True):
        return self.tokenizer(data)