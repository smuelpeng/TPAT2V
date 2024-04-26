from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from einops import rearrange

import gc
import tpa
from tpa.systems.base import BaseLossConfig, BaseSystem
from tpa.utils.typing import *
from tpa.utils.misc import time_recorder as tr
from tpa.utils.metrics import sim_matrix_training, sim_matrix_inference_stochastic, sim_matrix_inference_stochastic_light_allops, generate_embeds_per_video_id_stochastic, np_softmax
from tpa.modes.loss import CLIPLoss
from tpa.utils.metrics import t2v_metrics, v2t_metrics
from tpa.utils.misc import gen_log
from tqdm import tqdm

import time
import numpy as np


@dataclass
class TextMassLossConfig(BaseLossConfig):
    lambda_support_set: Any = 0.0

    
class TextMassTPA(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        loss: TextMassLossConfig = TextMassLossConfig()

        tokenizer_cls: str = ""
        tokenizer: dict = field(default_factory=dict)

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        post_processor_cls: str = ""
        post_processor: dict = field(default_factory=dict)

        pool_type: str = "mean"
        save_memory_mode: bool = False 
        stochasic_trials: int = 1
        DSL: bool = False
        metrics: str = "tpa.utils.metrics.t2v_metrics"


    cfg: Config

    def configure(self) -> None:
        self.tokenizer = tpa.find(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        self.backbone = tpa.find(self.cfg.backbone_cls)(self.cfg.backbone)
        self.post_processor = tpa.find(self.cfg.post_processor_cls)(self.cfg.post_processor)
        self.pooling_type = self.cfg.pool_type
        self.loss = CLIPLoss()
        self.metrics = tpa.find(self.cfg.metrics)()
        super().configure()

    
    def on_fit_start(self) -> None:
        return super().on_fit_start()   
    
    def forward(self, batch:Dict[str, Any]) -> Dict[str, Any]:
        if self.tokenizer is not None:
            batch['text'] = self.tokenizer(batch['text'], return_tensors='pt',
                                           padding=True, truncation=True)
        if self.training:
            text_embeds, video_embeds_pooled, text_embeds_stochastic, text_mean, text_log_var = self.backbone(batch, is_train=True)
            return {'text_embeds': text_embeds, 'video_embeds_pooled': video_embeds_pooled,
                    'text_embeds_stochastic': text_embeds_stochastic, 'text_mean': text_mean, 'text_log_var': text_log_var}
        else:
            text_embed, vid_embed, vid_embed_pooled, text_embed_stochastic = self.backbone(batch, return_all_frames=True, is_train=False)
            return {'text_embed': text_embed, 'vid_embed': vid_embed, 
                    'vid_embed_pooled': vid_embed_pooled, 
                    'text_embed_stochastic': text_embed_stochastic,
                    'vid_ids': batch['vid_ids']}
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        text_embeds_stochastic = outputs['text_embeds_stochastic']
        video_embeds_pooled = outputs['video_embeds_pooled']
        text_embeds = outputs['text_embeds']
        text_log_var = outputs['text_log_var']

        output = sim_matrix_training(text_embeds_stochastic, video_embeds_pooled, self.pooling_type)
        loss = self.loss(output, self.backbone.clip.logit_scale)
        
        # @WJM: support text embedding regulrization
        video_embeds_pooled_avg = torch.mean(video_embeds_pooled,dim=1).squeeze()
        pointer = video_embeds_pooled_avg - text_embeds
        text_support = pointer / pointer.norm(dim=-1, keepdim=True) * torch.exp(text_log_var) + text_embeds
        output_support = sim_matrix_training(text_support, video_embeds_pooled, self.pooling_type)
        loss_support = self.loss(output_support, self.backbone.clip.logit_scale)


        loss_all = loss + loss_support * self.cfg.loss.lambda_support_set

        return {"loss": loss_all}
    
    def compute_metric(self, batch, outputs):

        return
    
    def on_check_train(self, batch, outputs, **kwargs):
        # return super().on_check_train(batch, outputs, **kwargs)

        pass
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        return outputs


    def validation_end(self, validation_step_outputs):
        total_val_loss = 0.0
        text_embed_arr = []
        vid_embed_arr = []
        all_vid_ids = []        
        for out in validation_step_outputs:        
            text_embed = out['text_embed']
            vid_embed = out['vid_embed']
            text_embed_arr.append(text_embed)
            vid_embed_arr.append(vid_embed)
            all_vid_ids.append(out['vid_ids'])
        text_embeds = torch.cat(text_embed_arr)
        vid_embeds = torch.cat(vid_embed_arr)

        # Since we have all pairs, remove duplicate videos when there's multiple captions per video
        vid_embeds_per_video_id = {}
        for idx, v_id in enumerate(all_vid_ids):
            if v_id not in vid_embeds_per_video_id:
                vid_embeds_per_video_id[v_id] = vid_embeds[idx]

        vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])

        # Pool frames for inference once we have all texts and videos
        self.backbone.pool_frames.cpu()
        vid_embeds_pooled = self.backbone.pool_frames(text_embeds, vid_embeds)
        self.backbone.pool_frames.cuda()

        # build stochastic text embeds #########################################
        self.backbone.stochastic.cpu()
        start_selection_time = time.time()
        # initialize text_embeds_stochastic_allpairs: to avoid data leakage, break vid-txt dependence by dataloader
        text_embeds_stochastic_allpairs = torch.zeros(size=(vid_embeds.shape[0], text_embeds.shape[0], text_embeds.shape[1]))
        # @WJM: the principle is to use the query video to process text
        # sequential process to save memory:
        for (idx_vid, single_vid), single_vid_embed_pooled in tqdm(zip(enumerate(vid_embeds),vid_embeds_pooled)):

            single_vid_vec = single_vid.unsqueeze(0)
            # repeat as the same size of all texts
            single_vid_repeat = single_vid_vec.tile((text_embeds.shape[0], 1, 1)) # [bs_t, #F, dim]

            all_text_embed_stochstic = []
            for trial in range(self.cfg.stochasic_trials):
                all_text_embed_stochastic, _, _ = self.backbone.stochastic(text_embeds, single_vid_repeat) # [bs_t, dim]
                all_text_embed_stochstic.append(all_text_embed_stochastic)
            all_text_embed_stochstic_arr = torch.stack(all_text_embed_stochstic, dim=0) # [#trials, bs_t, dim]

            # normalization before compute cos-sim
            all_text_embed_stochstic_arr = all_text_embed_stochstic_arr / all_text_embed_stochstic_arr.norm(dim=-1, keepdim=True)
            single_vid_embed_pooled = single_vid_embed_pooled / single_vid_embed_pooled.norm(dim=-1, keepdim=True)

            # compute cos-sim
            sim_select = torch.sum(torch.mul(all_text_embed_stochstic_arr, single_vid_embed_pooled), dim=-1) # [#trial, bs_t]

            # find max cos, take idx
            max_indices = torch.argmax(sim_select, dim=0) # [bs_t]

            # select based on the idx
            selected_plane = torch.ones((all_text_embed_stochstic_arr.shape[1], all_text_embed_stochstic_arr.shape[2]))
            for i in range(all_text_embed_stochstic_arr.shape[1]):
                selected_plane[i, :] = all_text_embed_stochstic_arr[max_indices[i], i, :]
            text_embeds_stochastic_allpairs[idx_vid,:,:] = selected_plane

        end_selection_time = time.time()
        msg = (f'To compute all stochastic-text embeddings for the whole dataset, the time usage is {end_selection_time - start_selection_time}\n')
        # gen_log(model_path=self.cfg.model_path, log_name='log_trntst', msg=msg)
        self.backbone.stochastic.cuda()
        # finish build stochastic text embeds #########################################

        # @WJM: rm unnecessary tensor to release memory
        del text_embeds, vid_embeds
        gc.collect()

        text_embeds_per_video_id, vid_embeds_pooled_per_video_id = generate_embeds_per_video_id_stochastic(text_embeds_stochastic_allpairs,
                vid_embeds_pooled, all_vid_ids, self.pooling_type)

        # @WJM: rm unnecessary tensor to release memory
        del text_embeds_stochastic_allpairs, vid_embeds_pooled
        gc.collect()

        # @WJM: can use light implementation to avoid memory OOM:
        if self.cfg.save_memory_mode:
            start_sims = time.time()
            # gen_log(model_path=self.cfg.model_path, log_name='log_trntst', msg='Use sim_matrix_inference_stochastic_light()')
            sims = sim_matrix_inference_stochastic_light_allops(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, self.pooling_type, self.cfg.batch_size_split, self.cfg)
            end_sims = time.time()
            # gen_log(model_path=self.cfg.model_path, log_name='log_trntst', msg=f'batch size split = {self.cfg.batch_size_split}, sims compute time={end_sims-start_sims}')
        else:
            sims = sim_matrix_inference_stochastic(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, self.pooling_type)

        total_val_loss = total_val_loss / len(self.valid_data_loader)

        # add DSL
        if self.cfg.DSL:
            sims = sims * np_softmax(sims*100, axis=0)


        metrics = self.metrics
        res = metrics(sims)
        
        # Compute window metrics
        for m in res:
            self.window_metric[m].append(res[m])

        # Compute average of window metrics
        for m in self.window_metric:
            res[m + "-window"] = np.mean(self.window_metric[m])
        msg = (f"-----Val Epoch: {self.current_epoch}, dl: {self.true_global_step}/{self.true_current_epoch}-----\n",
                f"R@1: {res['R1']} (window: {res['R1-window']})\n", 
                f"R@5: {res['R5']} (window: {res['R5-window']})\n", 
                f"R@10: {res['R10']} (window: {res['R10-window']})\n",
                f"MedR: {res['MedR']} (window: {res['MedR-window']})\n",
                f"MeanR: {res['MeanR']} (window: {res['MeanR-window']})\n",
                )
        gen_log(model_path=self.cfg.model_path, log_name='log_trntst', msg=msg)
        # self.log(f'val', msg)
        # res['loss_val'] =  total_val_loss
        for k, v in res.items():
            self.log(f'val/{k}', v)
        return res
