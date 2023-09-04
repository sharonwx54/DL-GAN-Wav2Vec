# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum, auto
import math
import numpy as np
from typing import Tuple, List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import torchaudio.transforms as tat


from fairseq import checkpoint_utils, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel
from fairseq.modules import (
    SamePad,
    TransposeLast,
)

@dataclass
class SegmentationConfig(FairseqDataclass):
    subsample_rate: float = 0.25
    mean_pool: bool = True
    mean_pool_join: bool = False
    remove_zeros: bool = False


@dataclass
class Wav2vec_UConfig(FairseqDataclass):

    discriminator_kernel: int = 6
    discriminator_dilation: int = 1
    discriminator_dim: int = 384
    discriminator_causal: bool = True
    discriminator_linear_emb: bool = False
    discriminator_depth: int = 2
    discriminator_max_pool: bool = False
    discriminator_act_after_linear: bool = False
    discriminator_dropout: float = 0 #0.5
    discriminator_spectral_norm: bool = False
    discriminator_weight_norm: bool = False

    generator_kernel: int = 4
    generator_dilation: int = 1
    generator_stride: int = 1
    generator_bias: bool = False
    generator_dropout: float = 0.1

    blank_weight: float = 0
    blank_mode: str = "add"
    blank_is_sil: bool = False
    no_softmax: bool = False

    smoothness_weight: float = 0.5
    smoothing: float = 0.0
    smoothing_one_sided: bool = False
    gradient_penalty: float = 1.5
    probabilistic_grad_penalty_slicing: bool = False
    code_penalty: float = 4.0
    gumbel: bool = False
    hard_gumbel: bool = False
    temp: Tuple[float, float, float] = (2, 0.1, 0.99995)
    input_dim: int = 512

    segmentation: SegmentationConfig = SegmentationConfig()


class Segmenter(nn.Module):
    cfg: SegmentationConfig

    def __init__(self, cfg: SegmentationConfig):
        super().__init__()
        self.cfg = cfg
        self.subsample_rate = cfg.subsample_rate

    def pre_segment(self, dense_x, dense_padding_mask):
        return dense_x, dense_padding_mask

    def logit_segment(self, logits, padding_mask):
        return logits, padding_mask

class JoinSegmenter(Segmenter):
    def logit_segment(self, logits, padding_mask):
        preds = logits.argmax(dim=-1)

        if padding_mask.any():
            preds[padding_mask] = -1  # mark pad
        uniques = []

        bsz, tsz, csz = logits.shape

        for p in preds:
            uniques.append(
                p.cpu().unique_consecutive(return_inverse=True, return_counts=True)
            )

        new_tsz = max(u[0].numel() for u in uniques)
        new_logits = logits.new_zeros(bsz, new_tsz, csz)
        new_pad = padding_mask.new_zeros(bsz, new_tsz)

        for b in range(bsz):
            u, idx, c = uniques[b]
            keep = u != -1

            if self.cfg.remove_zeros:
                keep.logical_and_(u != 0)

            if self.training and not self.cfg.mean_pool_join:
                u[0] = 0
                u[1:] = c.cumsum(0)[:-1]
                m = c > 1
                r = torch.rand(m.sum())
                o = (c[m] * r).long()
                u[m] += o
                new_logits[b, : u.numel()] = logits[b, u]
            else:
                new_logits[b].index_add_(
                    dim=0, index=idx.to(new_logits.device), source=logits[b]
                )
                new_logits[b, : c.numel()] /= c.unsqueeze(-1).to(new_logits.device)

            new_sz = keep.sum()
            if not keep.all():
                kept_logits = new_logits[b, : c.numel()][keep]
                new_logits[b, :new_sz] = kept_logits

            if new_sz < new_tsz:
                pad = new_tsz - new_sz
                new_logits[b, -pad:] = 0
                new_pad[b, -pad:] = True

        return new_logits, new_pad

class Discriminator(nn.Module):
    def __init__(self, dim, cfg: Wav2vec_UConfig):
        super().__init__()

        inner_dim = cfg.discriminator_dim
        kernel = cfg.discriminator_kernel
        dilation = cfg.discriminator_dilation
        self.max_pool = cfg.discriminator_max_pool

        if cfg.discriminator_causal:
            padding = kernel - 1
        else:
            padding = kernel // 2

        def make_conv(in_d, out_d, k, p=0, has_dilation=True):
            conv = nn.Conv1d(
                in_d,
                out_d,
                kernel_size=k,
                padding=p,
                dilation=dilation if has_dilation else 1,
            )
            if cfg.discriminator_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            elif cfg.discriminator_weight_norm:
                conv = nn.utils.weight_norm(conv)
            return conv

        inner_net = [
            nn.Sequential(
                make_conv(inner_dim, inner_dim, kernel, padding),
                SamePad(kernel_size=kernel, causal=cfg.discriminator_causal),
                nn.Dropout(cfg.discriminator_dropout),
                nn.GELU(),
            )
            for _ in range(cfg.discriminator_depth - 1)
        ] + [
            make_conv(inner_dim, 1, kernel, padding, has_dilation=False),
            SamePad(kernel_size=kernel, causal=cfg.discriminator_causal),
        ]

        if cfg.discriminator_linear_emb:
            emb_net = [make_conv(dim, inner_dim, 1)]
        else:
            emb_net = [
                make_conv(dim, inner_dim, kernel, padding),
                SamePad(kernel_size=kernel, causal=cfg.discriminator_causal),
            ]

        if cfg.discriminator_act_after_linear:
            emb_net.append(nn.GELU())

        self.net = nn.Sequential(*emb_net, nn.Dropout(cfg.discriminator_dropout), *inner_net) # TODO: Write a nn.Sequential block of (*emb_net, dropout of discriminator_dropout, and *inner_net)

    def forward(self, x, padding_mask):
        x = x.transpose(1, 2)  # BTC -> BCT
        x = self.net(x)
        x = x.transpose(1, 2)
        x_sz = x.size(1)
        if padding_mask is not None and padding_mask.any() and padding_mask.dim() > 1:
            padding_mask = padding_mask[:, : x.size(1)]
            x[padding_mask] = float("-inf") if self.max_pool else 0
            x_sz = x_sz - padding_mask.sum(dim=-1)
        x = x.squeeze(-1)
        if self.max_pool:
            x, _ = x.max(dim=-1)
        else:
            x = x.sum(dim=-1)
            x = x / x_sz
        return x


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, cfg: Wav2vec_UConfig):
        super().__init__()

        self.cfg = cfg
        self.output_dim = output_dim
        self.stride = cfg.generator_stride
        self.dropout = nn.Dropout(cfg.generator_dropout)

        padding = cfg.generator_kernel // 2
        self.proj = nn.Sequential(
            TransposeLast(),
            nn.Conv1d(
                input_dim, output_dim,
                kernel_size=cfg.generator_kernel,
                stride=cfg.generator_stride,
                dilation=cfg.generator_dilation,
                padding=padding,
                bias=cfg.generator_bias,),
            # TODO: Call nn.Conv1d with parameters input_dim, output_dim, take the generator kernel size, stride, dilation, bias and padding
            TransposeLast(),
        )

    def forward(self, dense_x, tokens, dense_padding_mask):
        dense_x = self.dropout(dense_x)

        dense_x = self.proj(dense_x)
        if self.stride > 1:
            dense_padding_mask = dense_padding_mask[:, :: self.stride]

        if dense_padding_mask.size(1) != dense_x.size(1):
            new_padding = dense_padding_mask.new_zeros(dense_x.shape[:-1])
            diff = new_padding.size(1) - dense_padding_mask.size(1)
            assert (
                diff > 0
            ), f"{new_padding.shape}, {dense_padding_mask.shape}, {dense_x.shape}, {diff}"
            if diff > 0:
                new_padding[:, diff:] = dense_padding_mask
            else:
                assert diff < 0
                new_padding = dense_padding_mask[:, :diff]

            dense_padding_mask = new_padding

        result = {}

        token_x = None
        if tokens is not None:
            token_x = dense_x.new_zeros(tokens.numel(), self.output_dim)
            token_x.scatter_(1, tokens.view(-1, 1).long(), 1)
            token_x = token_x.view(tokens.shape + (self.output_dim,))

        result["dense_x"] = dense_x
        result["token_x"] = token_x
        result["dense_padding_mask"] = dense_padding_mask

        return result


class PermuteBlock(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)

class Wav2vec_U(nn.Module):

    def __init__(self, target_dict, cfg: Wav2vec_UConfig=Wav2vec_UConfig):
        super().__init__()

        self.cfg = cfg
        self.zero_index = target_dict.index("<SIL>") if "<SIL>" in target_dict else 0
        self.smoothness_weight = cfg.smoothness_weight

        output_size = len(target_dict)
        self.pad = target_dict.pad()
        self.eos = target_dict.eos()
        self.smoothing = cfg.smoothing
        self.smoothing_one_sided = cfg.smoothing_one_sided
        self.no_softmax = cfg.no_softmax
        self.gumbel = cfg.gumbel
        self.hard_gumbel = cfg.hard_gumbel
        self.last_acc = None

        self.gradient_penalty = cfg.gradient_penalty
        self.code_penalty = cfg.code_penalty
        self.blank_weight = cfg.blank_weight
        self.blank_mode = cfg.blank_mode
        self.blank_index = target_dict.index("<SIL>") if cfg.blank_is_sil else 0
        assert self.blank_index != target_dict.unk()

        self.discriminator =  Discriminator(output_size, cfg) # TODO: Call the Discriminator class with output_size and config

        self.pca_A = self.pca_b = None
        d = cfg.input_dim

        self.segmenter = JoinSegmenter(cfg.segmentation) # TODO: Call the JoinSegmenter class with the segmentation from config

        self.generator = Generator(d, output_size, cfg) # TODO: Call the Generator class with d, output_size and config
        self.augmentation = torch.nn.Sequential(
          tat.FrequencyMasking(freq_mask_param=100),
          tat.TimeMasking(time_mask_param=25)
        )
        self.add_noise = False # True

        self.max_temp, self.min_temp, self.temp_decay = cfg.temp
        self.curr_temp = self.max_temp
        self.update_num = 0
  
    def calc_gradient_penalty(self, real_data, fake_data):
        # TODO: implement gradient penalty
        # take the min value for batch and timestep size
        batch = min(real_data.size(0), fake_data.size(0))
        timesteps = min(real_data.size(1), fake_data.size(1))

        if self.cfg.probabilistic_grad_penalty_slicing:

            def get_slice(data, dim, target_size):

                size = data.size(dim)
                diff = size - target_size
                if diff <= 0:
                    return data

                start = np.random.randint(0, diff + 1)
                return data.narrow(dim=dim, start=start, length=target_size)

            real_data = get_slice(real_data, 0, batch)
            real_data = get_slice(real_data, 1, timesteps)
            fake_data = get_slice(fake_data, 0, batch)
            fake_data = get_slice(fake_data, 1, timesteps)

        else:
            real_data = real_data[:batch, :timesteps]
            fake_data = fake_data[:batch, :timesteps]

        alpha = torch.rand(real_data.size(0), 1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(real_data.device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self.discriminator(interpolates, None)

        gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=real_data.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradient_penalty = (gradients.norm(2, dim=1) - 1) ** 2
        return gradient_penalty

    def set_num_updates(self, num_updates):
        self.update_num = num_updates
        self.curr_temp = max(
            self.max_temp * self.temp_decay ** num_updates, self.min_temp
        )

    def discrim_step(self, num_updates):
        return num_updates % 2 == 1

    def get_groups_for_update(self):
        return "discriminator" if self.discrim_step(self.update_num) else "generator"

    def get_logits(
        self,
        net_output: Optional[Dict[str, List[Optional[torch.Tensor]]]],
        normalize: bool = False,
    ):
        logits = net_output["logits"]

        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., self.blank_index] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., self.blank_index] = self.blank_weight
            else:
                raise Exception(f"invalid blank mode {self.blank_mode}")

        padding = net_output["padding_mask"]
        if padding.any():
            logits[padding] = float("-inf")
            logits[padding][..., self.blank_index] = float("inf")

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)

        return logits.transpose(0, 1)

    def get_normalized_probs(
        self,
        net_output: Tuple[
            torch.Tensor, Optional[Dict[str, List[Optional[torch.Tensor]]]]
        ],
        log_probs: bool,
        sample: Optional[Dict[str, torch.Tensor]] = None,
    ):
        logits = self.get_logits(net_output)

        probs = super().get_normalized_probs(logits, log_probs, sample)
        # BTC -> TBC for ctc
        probs = probs.transpose(0, 1)
        return probs

    def normalize(self, dense_x):

        bsz, tsz, csz = dense_x.shape

        if dense_x.numel() == 0:
            raise Exception(dense_x.shape)
        _, k = dense_x.max(-1)
        hard_x = (
            dense_x.new_zeros(bsz * tsz, csz)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(-1, csz)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0) #TODO: Call torch.mean on hard_x after converting it to float and use dim == 0 
        code_perplexity = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        )

        avg_probs = torch.softmax(dense_x.reshape(-1, csz).float(), dim=-1).mean(dim=0)
        prob_perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        )

        if not self.no_softmax:
            if self.training and self.gumbel:
                dense_x = F.gumbel_softmax(
                    dense_x.float(), tau=self.curr_temp, hard=self.hard_gumbel
                ).type_as(dense_x)
            else:
                dense_x = dense_x.softmax(-1)

        return dense_x, code_perplexity, prob_perplexity

    def forward(
        self,
        features,
        padding_mask,
        random_label=None,
        dense_x_only=False,
        segment=True,
    ):
        
        if self.training:
            features = self.augmentation(features)
        """
        if self.add_noise:
            noise = np.random.randn(features.size(0), features.size(1), features.size(2))
            noise = torch.Tensor(noise).to(features.device)
            features = features+noise*0.01
        """
        if segment:
            features, padding_mask = self.segmenter.pre_segment(features, padding_mask)
        
        orig_size = features.size(0) * features.size(1) - padding_mask.sum()

        gen_result = self.generator(features, random_label, padding_mask)

        orig_dense_x, token_x = gen_result["dense_x"], gen_result["token_x"]
        orig_dense_padding_mask = gen_result["dense_padding_mask"]

        if segment:
            dense_x, dense_padding_mask = self.segmenter.logit_segment(
                orig_dense_x, orig_dense_padding_mask
            )
        else:
            dense_x = orig_dense_x
            dense_padding_mask = orig_dense_padding_mask

        dense_logits = dense_x
        prob_perplexity = None
        code_perplexity = None

        if not (self.no_softmax and dense_x_only):
            dense_x, code_perplexity, prob_perplexity = self.normalize(dense_logits)

        if dense_x_only or self.discriminator is None:
            return {
                "logits": dense_x,
                "padding_mask": dense_padding_mask,
            }

        token_padding_mask = random_label == self.pad

        dense_y = self.discriminator(dense_x, dense_padding_mask)
        token_y = self.discriminator(token_x, token_padding_mask)

        sample_size = features.size(0)

        d_step = self.discrim_step(self.update_num)

        fake_smooth = self.smoothing
        real_smooth = self.smoothing
        if self.smoothing_one_sided:
            fake_smooth = 0

        smoothness_loss = None
        code_pen = None

        if d_step:
            loss_dense = F.binary_cross_entropy_with_logits(
                dense_y,
                dense_y.new_ones(dense_y.shape) - fake_smooth,
                reduction="sum",
            )
            loss_token = F.binary_cross_entropy_with_logits(
                token_y,
                token_y.new_zeros(token_y.shape) + real_smooth,
                reduction="sum",
            )
            if self.training and self.gradient_penalty > 0:
                grad_pen = self.calc_gradient_penalty(token_x, dense_x) #TODO: Call the calc_gradient_penalty function you implemented with token_x and dense_x
                grad_pen = grad_pen.sum() * self.gradient_penalty
            else:
                grad_pen = None
        else:
            grad_pen = None
            loss_token = None
            loss_dense = F.binary_cross_entropy_with_logits(
                dense_y,
                dense_y.new_zeros(dense_y.shape) + fake_smooth,
                reduction="sum",
            )
            num_vars = dense_x.size(-1)
            if prob_perplexity is not None:
                code_pen = (num_vars - prob_perplexity) / num_vars
                code_pen = code_pen * sample_size * self.code_penalty

            if self.smoothness_weight > 0:
                smoothness_loss = F.mse_loss(
                    dense_logits[:, :-1], dense_logits[:, 1:], reduction="none"
                )
                smoothness_loss[dense_padding_mask[:, 1:]] = 0
                smoothness_loss = (
                    smoothness_loss.mean() * sample_size * self.smoothness_weight
                )

        result = {
            "losses": {
                "grad_pen": grad_pen,
                "code_pen": code_pen,
                "smoothness": smoothness_loss,
            },
            "temp": self.curr_temp,
            "code_ppl": code_perplexity,
            "prob_ppl": prob_perplexity,
            "d_steps": int(d_step),
            "sample_size": sample_size,
        }

        suff = "_dis" if d_step else "_gen"
        result["losses"]["dense" + suff] = loss_dense
        result["losses"]["token" + suff] = loss_token
        
        return result
