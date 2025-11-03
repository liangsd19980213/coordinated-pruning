from abc import ABC, abstractmethod
from pruning.metric_logger import MetricLogger
from pruning.utils import nanstd
import logging, os

import torch

metric_logger = MetricLogger()
logger = logging.getLogger(__name__)


class AbstractAttenPruner(ABC, torch.nn.Module):

    def __init__(self, idx, alpha):
        super().__init__()
        self.idx = idx
        self.alpha = alpha

    @abstractmethod
    def get_pruning_mask(self, layer_attention_probes: torch.tensor, mask: torch.tensor) -> torch.tensor:
        raise NotImplementedError("Pruning method not implemeted.")

    def forward(self, layer_attention_probes: torch.tensor, mask: torch.tensor):
        return self.get_pruning_mask(layer_attention_probes, mask)


class IQRPruner(AbstractAttenPruner):
    def get_pruning_mask(self, layer_attention_probes: torch.tensor, mask: torch.tensor) -> torch.tensor:
        _device = mask.device
        # Squeeze the unnecessary dimensions
        mask_squeezed = mask.clone().squeeze(1).squeeze(1)  # Resulting shape: (bs, seq_len)
        # Take the mean across all heads -> Resulting shape: (bs, seq_len, seq_len)
        attention_probes = layer_attention_probes.clone().mean(dim=1)
        # Get position of SEP token
        sep_pos = (mask_squeezed.sum(dim=1) - 1).unsqueeze(1)
        # CLS position is 0, and PAD pos is mask_squeezed[sep_pos+1:]
        # Take Column-wise mean -> Resulting shape: (bs, seq_len)
        scores = attention_probes.mean(dim=1)
        # Set CLS, SEP, and PAD to NaNs
        scores[:, 0] = float('nan')  # CLS
        scores.scatter_(dim=-1, index=sep_pos.long(), value=float('nan'))  # SEP

        # PAD
        seq_len = mask_squeezed.shape[1]
        bs = mask_squeezed.shape[0]
        indices = sep_pos + 1
        sep_pos = torch.clamp(sep_pos, min=0, max=seq_len - 1)
        _upper_bound = torch.full((bs, 1), seq_len, device=_device)
        # _upper_bound = torch.tensor([[seq_len] * bs], device=_device).view((bs, 1))
        interval_bounds = torch.cat((indices, _upper_bound), 1)
        _indices = torch.arange(seq_len, device=_device)[None, :]
        lower_bounds = interval_bounds[:, 0, None]
        upper_bounds = interval_bounds[:, 1, None]
        interval_mask = ((_indices >= lower_bounds) & (_indices <= upper_bounds))
        scores[interval_mask] = float('nan')
        # Calculate Q1 and Q3, ignoring the NaNs
        # q2 = torch.nanquantile(scores, 0.25, dim=1, keepdim=True)
        # q3 = torch.nanquantile(scores, 0.75, dim=1, keepdim=True)
        _mean = torch.nanmean(scores, dim=1, keepdim=True)
        _std = nanstd(scores, 1).unsqueeze(1)
        _max = _mean + (self.alpha * _std)
        _min = _mean - (self.alpha * _std)
        updated_mask = torch.zeros_like(mask_squeezed)
        intermediate_mask = (
                    ((scores >= _min) & (~torch.isnan(scores))) & (((scores <= _max) & (~torch.isnan(scores)))))
        indices = intermediate_mask
        updated_mask[indices] = 1
        # Set CLS and SEP to 1
        updated_mask[:, 0] = 1
        updated_mask.scatter_(dim=-1, index=sep_pos.long(), value=1)  # SEP
        return updated_mask


class StaticPruner(AbstractAttenPruner):
    def __init__(self, idx, alpha, pruning_ratio=0.6):
        super().__init__(idx, alpha)
        self.pruning_ratio = pruning_ratio

    def get_pruning_mask(self, layer_attention_probes: torch.tensor, mask: torch.tensor) -> torch.tensor:
        # 计算平均注意力分数
        attention_scores = layer_attention_probes.clone().mean(dim=1).mean(dim=1)  # [bs, seq_len]
        mask_squeezed = mask.clone().squeeze(1).squeeze(1)  # [bs, seq_len]
        bs, seq_len = mask_squeezed.shape

        # 设置 CLS 和 SEP 的分数为 NaN，避免剪掉
        sep_pos = (mask_squeezed.sum(dim=1) - 1).unsqueeze(1)
        attention_scores[:, 0] = float('nan')  # CLS
        attention_scores.scatter_(dim=-1, index=sep_pos.long(), value=float('nan'))  # SEP

        updated_mask = torch.zeros_like(mask_squeezed)
        k = (mask_squeezed.sum(dim=1) * (1 - self.pruning_ratio)).long()  # 每个样本保留的 token 数

        for i in range(bs):
            valid_scores = attention_scores[i]
            valid_indices = (~torch.isnan(valid_scores)).nonzero(as_tuple=True)[0]
            scores = valid_scores[valid_indices]
            if scores.numel() == 0 or k[i] <= 0:
                updated_mask[i, :] = mask_squeezed[i]
                continue

            # 取 top-k 的 token 分数索引
            topk_indices = torch.topk(scores, k[i], largest=True).indices
            preserved_token_indices = valid_indices[topk_indices]
            updated_mask[i, preserved_token_indices] = 1

            # 确保保留 CLS 和 SEP
            updated_mask[i, 0] = 1
            updated_mask[i, sep_pos[i, 0]] = 1

        return updated_mask
